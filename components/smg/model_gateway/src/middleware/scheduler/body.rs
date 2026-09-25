//! Response-body wrapper that ties a [`SchedulerPermit`] to the response
//! stream: it marks TTFT on the first data frame and releases the slot
//! when the body finishes draining (or is dropped).

use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use axum::{body::Body, http::StatusCode};
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame};

use super::{engine::SchedulerPermit, output_tokens::observed_output_tokens, SettlementKind};
use crate::middleware::admission_metrics::AdmissionActiveGuard;

const FAIR_SHARE_RESPONSE_TAIL_LIMIT: usize = 64 * 1024;

/// Wraps a response [`Body`], holding the request's [`SchedulerPermit`]
/// for the lifetime of the stream.
///
/// - **TTFT.** On the first *data* frame, marks time-to-first-byte via the
///   permit's CAS. If the scheduler already won the preemption CAS
///   (`try_mark_first_byte` returns `false`), the request was preempted in
///   the narrow gap before its first byte — we end the stream
///   (`Poll::Ready(None)`) rather than emit a byte the client should never
///   see. The handler's cancel token has fired, so it is unwinding anyway.
///   Only data frames mark TTFT; trailer-only responses never do.
/// - **Slot release.** When this body drops (stream complete, client
///   disconnected, or handler unwound), the `permit` field drops with it,
///   and `SchedulerPermit::drop` releases the slot back to the scheduler
///   and unregisters the inflight handle. No explicit `Drop` is needed —
///   the field's natural drop carries the invariant.
pub struct SchedulerGuardBody {
    inner: Body,
    permit: SchedulerPermit,
    active_guard: AdmissionActiveGuard,
    status_code: StatusCode,
    completed: bool,
    /// Latch so the TTFT CAS is attempted only once (on the first data
    /// frame); subsequent frames skip the check entirely.
    ttft_marked: bool,
    /// Set once the stream is force-ended by a lost TTFT/preempt CAS. Keeps
    /// the terminal state idempotent across re-polls and keeps
    /// `is_end_stream` consistent with `poll_frame`: the inner body still
    /// has data, but this wrapper has already signalled `None`.
    terminated: bool,
    response_tail: Option<VecDeque<u8>>,
    fair_share_settled: bool,
}

impl SchedulerGuardBody {
    pub(crate) fn new(
        inner: Body,
        permit: SchedulerPermit,
        active_guard: AdmissionActiveGuard,
        status_code: StatusCode,
    ) -> Self {
        let completed = inner.is_end_stream();
        let response_tail = permit
            .has_fair_share_reservation()
            .then(|| VecDeque::with_capacity(FAIR_SHARE_RESPONSE_TAIL_LIMIT));
        let fair_share_settled = response_tail.is_none();
        Self {
            inner,
            permit,
            active_guard,
            status_code,
            completed,
            ttft_marked: false,
            terminated: false,
            response_tail,
            fair_share_settled,
        }
    }

    fn append_response_tail(&mut self, data: &[u8]) {
        let Some(response_tail) = self.response_tail.as_mut() else {
            return;
        };
        if data.len() >= FAIR_SHARE_RESPONSE_TAIL_LIMIT {
            response_tail.clear();
            response_tail.extend(
                data[data.len().saturating_sub(FAIR_SHARE_RESPONSE_TAIL_LIMIT)..]
                    .iter()
                    .copied(),
            );
            return;
        }
        let overflow = response_tail
            .len()
            .saturating_add(data.len())
            .saturating_sub(FAIR_SHARE_RESPONSE_TAIL_LIMIT);
        for _ in 0..overflow {
            response_tail.pop_front();
        }
        response_tail.extend(data.iter().copied());
    }

    fn settle_completed_response(&mut self) {
        if self.fair_share_settled {
            return;
        }
        let observed = self
            .response_tail
            .as_mut()
            .and_then(|tail| observed_output_tokens(tail.make_contiguous()));
        let kind = if observed.is_some() {
            SettlementKind::Observed
        } else {
            SettlementKind::MissingUsage
        };
        self.permit.settle_output_tokens(observed, kind);
        self.fair_share_settled = true;
    }
}

impl Drop for SchedulerGuardBody {
    fn drop(&mut self) {
        if self.completed {
            self.settle_completed_response();
            self.active_guard.record_outcome(self.status_code.as_u16());
        } else {
            self.active_guard.record_interrupted();
        }
    }
}

impl http_body::Body for SchedulerGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `UnsyncBoxBody`, which is `Unpin`; the other fields are
        // `Unpin` too, so `get_mut` + `Pin::new(&mut inner)` is sound
        // (mirrors `TokenGuardBody` in `concurrency.rs`).
        let this = self.get_mut();
        // Stay ended once we've force-ended; never poll the inner body again.
        if this.terminated {
            return Poll::Ready(None);
        }
        let polled = Pin::new(&mut this.inner).poll_frame(cx);

        if !this.ttft_marked {
            if let Poll::Ready(Some(Ok(frame))) = &polled {
                if frame.is_data() {
                    if this.permit.try_mark_first_byte() {
                        this.ttft_marked = true;
                    } else {
                        // Lost the TTFT race to a concurrent preemption
                        // CAS. Terminate the stream cleanly.
                        this.terminated = true;
                        this.completed = false;
                        return Poll::Ready(None);
                    }
                }
            }
        }
        if let Poll::Ready(Some(Ok(frame))) = &polled {
            if let Some(data) = frame.data_ref() {
                this.append_response_tail(data);
            }
        }
        match &polled {
            Poll::Ready(None) => {
                this.completed = true;
                this.settle_completed_response();
            }
            Poll::Ready(Some(Err(_))) => this.completed = false,
            _ => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        // Force-end takes precedence: once we've returned `None` on a lost
        // CAS, report the stream ended even though the inner body has not.
        self.terminated || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use http_body_util::BodyExt;
    use smg_auth::RequestId;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        middleware::scheduler::{
            AdmitOutcome, Class, ClassConfig, FairShareConfig, GlobalFairShare, PriorityScheduler,
            PrioritySchedulerYaml, SchedulerSettings,
        },
        tenant::TenantKey,
    };

    fn scheduler() -> Arc<PriorityScheduler> {
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap();
        // Default reservations sum to 160 (Interactive 128 + System 32),
        // so capacity must be >= 160 for new() to accept them.
        PriorityScheduler::new(&settings, 256).unwrap()
    }

    fn permit(sched: &Arc<PriorityScheduler>, id: &str) -> SchedulerPermit {
        sched
            .acquire_inflight(Class::Default, RequestId(id.to_string()))
            .expect("slot available")
    }

    fn guarded(inner: Body, permit: SchedulerPermit) -> SchedulerGuardBody {
        SchedulerGuardBody::new(inner, permit, AdmissionActiveGuard::new(), StatusCode::OK)
    }

    fn fair_scheduler(
        default_output_tokens: u32,
    ) -> (Arc<PriorityScheduler>, Arc<GlobalFairShare>, TenantKey, u64) {
        let mut classes = HashMap::new();
        for class in Class::ALL {
            let mut config = ClassConfig::default_for(class);
            config.reserved_floor = 0;
            config.reserved_per_slot = 0.0;
            config.queue_size = 8;
            classes.insert(class, config);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            fair_share: Some(FairShareConfig {
                default_weight: 1.0,
                default_output_tokens,
                trust_output_token_estimate_header: false,
                trust_request_model_header: false,
                tenant_weights: HashMap::from([("header:alice".to_string(), 1.0)]),
                model_profiles: HashMap::new(),
            }),
            ..Default::default()
        };
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        let ledger = Arc::new(GlobalFairShare::from_settings(&settings).unwrap());
        let scheduler =
            PriorityScheduler::new_with_fair_share(&settings, 1, Some(Arc::clone(&ledger)))
                .unwrap();
        let scope_id = scheduler
            .fair_share_scope_for_test()
            .expect("fair-share scheduler has a scope");
        (scheduler, ledger, TenantKey::new("header:alice"), scope_id)
    }

    async fn fair_permit(
        scheduler: &Arc<PriorityScheduler>,
        tenant: &TenantKey,
        id: &str,
        estimate: u32,
    ) -> SchedulerPermit {
        let outcome = scheduler
            .admit_for_tenant(
                Class::Default,
                RequestId(id.to_string()),
                CancellationToken::new(),
                tenant.clone(),
                estimate,
            )
            .await;
        let AdmitOutcome::Admitted(permit) = outcome else {
            panic!("request should be admitted");
        };
        permit
    }

    #[tokio::test]
    async fn test_first_data_frame_marks_ttft() {
        let sched = scheduler();
        let p = permit(&sched, "req-ttft");
        let handle = Arc::clone(p.handle());
        let mut guarded = guarded(Body::from("hello world"), p);

        assert!(handle.is_preemptible(), "pre-TTFT before first frame");
        // Drive the first frame (SchedulerGuardBody is Unpin).
        let frame = guarded.frame().await.expect("a frame").expect("ok frame");
        assert!(frame.is_data());
        assert!(
            !handle.is_preemptible(),
            "first data frame must mark TTFT (no longer preemptible)"
        );
        // A request that has emitted its first byte must reject preemption.
        assert!(!handle.try_mark_preempted());
    }

    #[tokio::test]
    async fn test_preempted_before_first_byte_ends_stream() {
        let sched = scheduler();
        let p = permit(&sched, "req-preempt");
        let handle = Arc::clone(p.handle());
        // Scheduler wins the preempt CAS before any byte is polled.
        assert!(handle.try_mark_preempted());

        let mut guarded = guarded(Body::from("should not surface"), p);
        // First poll must terminate the stream rather than yield the data.
        let next = guarded.frame().await;
        assert!(
            next.is_none(),
            "preempted body must end the stream on first frame"
        );
        // is_end_stream must agree with the forced None (consistency on the
        // CAS-loss path), and a re-poll must stay ended.
        assert!(
            http_body::Body::is_end_stream(&guarded),
            "force-ended stream must report is_end_stream() == true"
        );
        assert!(
            guarded.frame().await.is_none(),
            "re-polling a force-ended stream must stay ended"
        );
    }

    #[tokio::test]
    async fn test_trailer_only_response_never_marks_ttft() {
        let sched = scheduler();
        let p = permit(&sched, "req-empty");
        let handle = Arc::clone(p.handle());
        // Empty body: no data frames at all.
        let guarded = guarded(Body::empty(), p);
        let _ = guarded.collect().await; // exhaust
        assert!(
            handle.is_preemptible(),
            "no data frame emitted → TTFT never marked"
        );
    }

    #[tokio::test]
    async fn test_drop_releases_slot() {
        let sched = scheduler();
        assert_eq!(sched.inflight_for_test(Class::Default), 0);
        let p = permit(&sched, "req-drop");
        assert_eq!(sched.inflight_for_test(Class::Default), 1);
        let guarded = guarded(Body::from("x"), p);
        drop(guarded);
        assert_eq!(
            sched.inflight_for_test(Class::Default),
            0,
            "dropping the guarded body releases the slot"
        );
    }

    #[tokio::test]
    async fn disabled_mode_never_allocates_a_response_tail() {
        let sched = scheduler();
        let p = permit(&sched, "disabled-tail");
        let mut guarded = guarded(Body::from("hello"), p);
        assert!(guarded.response_tail.is_none());
        assert!(guarded.frame().await.unwrap().unwrap().is_data());
        assert!(guarded.response_tail.is_none());
    }

    #[tokio::test]
    async fn terminal_usage_settles_actual_output_tokens() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "usage", 100).await;
        let guarded = SchedulerGuardBody::new(
            Body::from(r#"{"usage":{"completion_tokens":37}}"#),
            permit,
            AdmissionActiveGuard::new(),
            StatusCode::OK,
        );
        guarded.collect().await.unwrap();

        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 37);
    }

    #[tokio::test]
    async fn completed_sse_usage_settles_actual_output_tokens() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "sse-usage", 100).await;
        let body = Body::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
             data: {\"choices\":[],\"usage\":{\"completion_tokens\":19}}\n\n\
             data: [DONE]\n\n",
        );
        guarded(body, permit).collect().await.unwrap();
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 19);
    }

    #[tokio::test]
    async fn split_sse_frames_are_reassembled_for_terminal_usage() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "split-sse", 100).await;
        let frames: Vec<Result<Bytes, std::io::Error>> = [
            "data: {\"choices\":[]",
            ",\"usage\":{\"completion_",
            "tokens\":23}}\n\n",
            "data: [DONE]\n\n",
        ]
        .into_iter()
        .map(|chunk| Ok(Bytes::from(chunk)))
        .collect();
        let body = Body::from_stream(tokio_stream::iter(frames));
        guarded(body, permit).collect().await.unwrap();
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 23);
    }

    #[tokio::test]
    async fn client_drop_before_usage_keeps_the_provisional_charge() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "client-drop", 100).await;
        let mut body = guarded(Body::from("data: {\"choices\":[]}\n\n"), permit);
        assert!(body.frame().await.unwrap().unwrap().is_data());
        drop(body);
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 100);
    }

    #[tokio::test]
    async fn backend_error_keeps_the_provisional_charge() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "backend-error", 100).await;
        let frames = vec![
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: partial\n\n")),
            Err(std::io::Error::other("backend failed")),
        ];
        let mut body = guarded(Body::from_stream(tokio_stream::iter(frames)), permit);
        assert!(body.frame().await.unwrap().unwrap().is_data());
        assert!(body.frame().await.unwrap().is_err());
        drop(body);
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 100);
    }

    #[tokio::test]
    async fn permit_settlement_and_drop_are_idempotent() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let mut permit = fair_permit(&scheduler, &tenant, "double-settle", 100).await;
        permit.settle_output_tokens(Some(31), SettlementKind::Observed);
        permit.settle_output_tokens(Some(999), SettlementKind::Observed);
        drop(permit);
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 31);
    }

    #[tokio::test]
    async fn many_small_frames_stay_bounded_and_parse_usage_after_ring_wrap() {
        let (scheduler, ledger, tenant, scope_id) = fair_scheduler(100);
        let permit = fair_permit(&scheduler, &tenant, "ring-wrap", 100).await;
        let mut frames: Vec<Result<Bytes, std::io::Error>> = (0..9_000)
            .map(|_| Ok(Bytes::from_static(b"data: filler\n\n")))
            .collect();
        for chunk in [
            "data: {\"choices\":[],\"usage\":{\"comple",
            "tion_tokens\":47}}\n\n",
            "data: [DONE]\n\n",
        ] {
            frames.push(Ok(Bytes::from(chunk)));
        }

        let mut body = guarded(Body::from_stream(tokio_stream::iter(frames)), permit);
        let mut frame_count = 0;
        while let Some(frame) = body.frame().await {
            frame.unwrap();
            frame_count += 1;
            assert!(
                body.response_tail
                    .as_ref()
                    .is_some_and(|tail| tail.len() <= FAIR_SHARE_RESPONSE_TAIL_LIMIT),
                "ring buffer must stay within its fixed cap"
            );
        }
        assert_eq!(frame_count, 9_003);
        assert_eq!(
            body.response_tail.as_ref().map(VecDeque::len),
            Some(FAIR_SHARE_RESPONSE_TAIL_LIMIT)
        );
        drop(body);
        assert_eq!(ledger.snapshot(scope_id, &tenant).0, 47);
    }
}
