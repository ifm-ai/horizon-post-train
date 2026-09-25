//! `priority_admission_middleware`: the axum layer that runs the priority
//! scheduler on protected routes when it is enabled.
//!
//! Pipeline position (route_layer order): runs after tenant resolution
//! (so `RouteRequestMeta.tenant_key` is in extensions for the clamp) and
//! before the handler. On admission it inserts the permit's cancel token
//! into request extensions (so long-running handlers can `select!` against
//! it for preemption) and wraps the response body in `SchedulerGuardBody`
//! (TTFT marking + slot release).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::Next,
    response::Response,
};
use smg_auth::RequestId;
use tokio_util::sync::CancellationToken;
use tracing::trace;

use super::{
    capacity_credit::CapacityCreditError,
    capacity_credit_api::{
        capacity_credit_error_response, capacity_credit_required_response,
        presented_capacity_credit,
    },
    fair_share::FairShareProfile,
    metrics as sched_metrics,
    state::SchedulerState,
    AdmitOutcome, Class, GlobalFairShare, LocalAdaptiveRejection, RejectionReason, SchedulerError,
    SchedulerGuardBody, SchedulerPermit, HEADER_X_SMG_PREEMPTED, OUTPUT_TOKEN_ESTIMATE_HEADER,
    PRIORITY_HEADER,
};
use crate::{
    middleware::{
        admission_metrics::{AdmissionActiveGuard, AdmissionPendingGuard},
        RouteRequestMeta,
    },
    observability::metrics::{metrics_labels, Metrics},
    tenant::TenantKey,
};

/// Monotonic source of registry keys for admitted requests. Each admission
/// gets a unique key, so the inflight registry can never be clobbered by a
/// duplicate client-supplied `x-request-id` (the collision hazard flagged
/// in earlier review). The client request id is still used for logging.
static NEXT_ADMISSION_ID: AtomicU64 = AtomicU64::new(0);

fn next_registry_id() -> RequestId {
    let n = NEXT_ADMISSION_ID.fetch_add(1, Ordering::Relaxed);
    RequestId(format!("sched-{n}"))
}

fn consume_local_adaptive_rejection_marker(
    response: &mut Response,
    permit: &mut SchedulerPermit,
) -> bool {
    let rejected = response
        .extensions_mut()
        .remove::<LocalAdaptiveRejection>()
        .is_some();
    if rejected {
        permit.cancel_fair_share_reservation();
    }
    rejected
}

/// The class a request resolves to, plus what it asked for.
struct ResolvedPriority {
    /// Post-clamp class the request is admitted under.
    effective: Class,
    /// Class parsed from the header before the tenant clamp.
    requested: Class,
    /// Header carried a non-empty value that didn't name a known class.
    unknown: bool,
}

/// Resolve the effective class: parse the priority header, then clamp it
/// down to the tenant's configured `max_class` (a low-tier tenant cannot
/// self-promote by setting the header). `min` is the clamp because of the
/// `Ord` derive on `Class`.
fn resolve_priority(
    req: &Request<Body>,
    state: &SchedulerState,
    tenant: &TenantKey,
) -> ResolvedPriority {
    let raw = req
        .headers()
        .get(PRIORITY_HEADER)
        .and_then(|h| h.to_str().ok());
    let requested = raw.map(Class::parse_header).unwrap_or(Class::Default);
    // Unknown = present, non-empty, not "default", yet still parsed to
    // Default (i.e. an unrecognized value silently downgraded).
    let unknown = raw.map(str::trim).is_some_and(|v| {
        !v.is_empty() && !v.eq_ignore_ascii_case("default") && requested == Class::Default
    });
    let max_class = state.resolver.policy(tenant).max_class;
    ResolvedPriority {
        effective: requested.min(max_class),
        requested,
        unknown,
    }
}

/// Map an admit rejection to an `admit_total` outcome label.
fn rejection_outcome(reason: RejectionReason) -> &'static str {
    match reason {
        RejectionReason::QueueFull => sched_metrics::outcome::REJECTED_QUEUE_FULL,
        RejectionReason::QueueTimeout => sched_metrics::outcome::REJECTED_QUEUE_TIMEOUT,
        RejectionReason::Preempted => sched_metrics::outcome::PREEMPTED,
        RejectionReason::ClientCancelled => sched_metrics::outcome::CLIENT_CANCELLED,
    }
}

fn output_token_estimate(headers: &HeaderMap, ledger: &GlobalFairShare) -> u32 {
    let configured_default = ledger.default_output_tokens();
    let header = headers.get(OUTPUT_TOKEN_ESTIMATE_HEADER);
    if !ledger.trusts_output_token_estimate_header() {
        if header.is_some() {
            sched_metrics::record_fair_share_fallback("untrusted_estimate");
        }
        return configured_default;
    }
    match header {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                sched_metrics::record_fair_share_fallback("invalid_estimate");
                configured_default
            }),
        None => {
            sched_metrics::record_fair_share_fallback("missing_estimate");
            configured_default
        }
    }
}

async fn run_admitted_request(
    mut req: Request<Body>,
    next: Next,
    mut permit: SchedulerPermit,
    class: Class,
    requested_class: Class,
    tenant: &TenantKey,
    partition: &super::state::SchedulerPartition,
) -> Response {
    Metrics::record_http_admission_admitted();
    let active_guard = AdmissionActiveGuard::new();
    req.extensions_mut().insert(permit.cancel_token());
    if let Some(proof) = permit.fair_share_admission_proof() {
        let Some(meta) = req.extensions().get::<RouteRequestMeta>().cloned() else {
            permit.cancel_fair_share_reservation();
            Metrics::record_http_admission_rejected();
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            return response;
        };
        req.extensions_mut().insert(meta.with_extension(proof));
    }
    let mut response = next.run(req).await;
    // A local adaptive rejection happened before backend work began. Release
    // the held slot normally, but erase the provisional output-token charge so
    // a retry does not accumulate fair-share debt.
    consume_local_adaptive_rejection_marker(&mut response, &mut permit);
    let outcome = if response.headers().contains_key(HEADER_X_SMG_PREEMPTED) {
        sched_metrics::outcome::PREEMPTED
    } else {
        sched_metrics::outcome::ADMITTED
    };
    sched_metrics::record_admit(class, outcome);
    sched_metrics::record_partition_admit(&partition.name, class, outcome);
    trace!(
        scheduler.class = class.as_str(),
        scheduler.requested_class = requested_class.as_str(),
        scheduler.tenant = %tenant,
        scheduler.partition = %partition.name,
        scheduler.admit_outcome = outcome,
        "scheduler admission decision"
    );
    let status_code = response.status();
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(SchedulerGuardBody::new(
            body,
            permit,
            active_guard,
            status_code,
        )),
    )
}

pub async fn priority_admission_middleware(
    State(state): State<Arc<SchedulerState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    Metrics::record_http_admission_received();
    let mut pending_guard = AdmissionPendingGuard::new();
    let tenant = req
        .extensions()
        .get::<RouteRequestMeta>()
        .map(|m| m.tenant_key().clone())
        .unwrap_or_else(|| TenantKey::new("anonymous"));
    let route_request_id = req
        .extensions()
        .get::<RouteRequestMeta>()
        .map(RouteRequestMeta::request_charge_id);

    // A credit already owns a scheduler slot. Redeem it before the sibling
    // RPS check so any local rejection can drop the held permit immediately
    // instead of stranding capacity until the credit TTL expires.
    let held_credit = match presented_capacity_credit(req.headers(), &tenant, &state) {
        Ok(Some(presented)) => {
            let Some(route_request_id) = route_request_id else {
                Metrics::record_http_admission_rejected();
                pending_guard.resolve();
                return capacity_credit_error_response(CapacityCreditError::InternalState, None);
            };
            let Some(registry) = state.capacity_credits() else {
                Metrics::record_http_admission_rejected();
                pending_guard.resolve();
                return capacity_credit_error_response(CapacityCreditError::Unknown, None);
            };
            match registry.redeem(&presented.token, &presented.binding) {
                Ok(mut redeemed) => {
                    if !redeemed
                        .payload
                        .bind_redeemed_capacity_credit(&redeemed.binding, route_request_id)
                    {
                        Metrics::record_http_admission_rejected();
                        pending_guard.resolve();
                        return capacity_credit_error_response(
                            CapacityCreditError::InternalState,
                            None,
                        );
                    }
                    Some((redeemed.payload, presented.partition))
                }
                Err(error) => {
                    Metrics::record_http_admission_rejected();
                    pending_guard.resolve();
                    return capacity_credit_error_response(error, None);
                }
            }
        }
        Ok(None) if state.capacity_credit_required() => {
            Metrics::record_http_admission_rejected();
            pending_guard.resolve();
            return capacity_credit_required_response();
        }
        Ok(None) => None,
        Err(error) => {
            Metrics::record_http_admission_rejected();
            pending_guard.resolve();
            return capacity_credit_error_response(error, None);
        }
    };

    // RPS sibling check (only set when an explicit per-second limit is
    // configured). Checked before admission so a rejected request never
    // acquires a new slot. A redeemed credit already owns its slot and is
    // dropped here on rejection. Tokens are not returned; refill is time-based.
    // Tracked by smg_http_rate_limit_total, so not double-counted here.
    if let Some(bucket) = &state.rate_limiter {
        if bucket.try_acquire(1.0).is_err() {
            drop(held_credit);
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_http_admission_rejected();
            pending_guard.resolve();
            return crate::rate_limit::rejection_response(bucket.retry_after_secs(1.0));
        }
    }

    if let Some((held_permit, partition)) = held_credit {
        pending_guard.resolve();
        return run_admitted_request(
            req,
            next,
            held_permit.into_permit(),
            Class::Default,
            Class::Default,
            &tenant,
            &partition,
        )
        .await;
    }

    let resolved = resolve_priority(&req, &state, &tenant);
    let class = resolved.effective;

    if resolved.unknown {
        sched_metrics::record_unknown_priority(tenant.as_str());
    }
    if class < resolved.requested {
        sched_metrics::record_clamp(resolved.requested, class, tenant.as_str());
    }

    let partition = state.partition_for(req.headers());
    let request_id = next_registry_id();

    // NOTE: client-disconnect detection during the queue wait is not yet
    // wired (axum does not surface it to a middleware pre-`next.run`), so we
    // pass a fresh token; queued waits are bounded by `queue_timeout`. A
    // real disconnect drops the response future (and the SchedulerGuardBody)
    // once admitted, releasing the slot.
    let cancel = CancellationToken::new();

    let (estimated_output_tokens, fair_share_profile) = match partition.scheduler.fair_share() {
        Some(ledger) => (
            output_token_estimate(req.headers(), ledger),
            state.fair_share_profile_for(req.headers(), ledger),
        ),
        None => (1, FairShareProfile::Global),
    };

    match partition
        .scheduler
        .admit_for_tenant_profile(
            class,
            request_id,
            cancel,
            tenant.clone(),
            fair_share_profile,
            estimated_output_tokens,
        )
        .await
    {
        AdmitOutcome::Admitted(permit) => {
            pending_guard.resolve();
            run_admitted_request(
                req,
                next,
                permit,
                class,
                resolved.requested,
                &tenant,
                &partition,
            )
            .await
        }
        AdmitOutcome::Rejected(reason) => {
            Metrics::record_http_admission_rejected();
            pending_guard.resolve();
            let outcome = rejection_outcome(reason);
            sched_metrics::record_admit(class, outcome);
            sched_metrics::record_partition_admit(&partition.name, class, outcome);
            trace!(
                scheduler.class = class.as_str(),
                scheduler.requested_class = resolved.requested.as_str(),
                scheduler.tenant = %tenant,
                scheduler.partition = %partition.name,
                scheduler.admit_outcome = outcome,
                "scheduler admission decision"
            );
            SchedulerError::from(reason)
                .into_response_with_retry_after(partition.scheduler.retry_after_secs(class))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::http::StatusCode;

    use super::*;
    use crate::middleware::scheduler::{
        ClassConfig, FairShareConfig, PriorityScheduler, PrioritySchedulerYaml, SchedulerSettings,
    };

    fn ledger(trust_header: bool) -> GlobalFairShare {
        GlobalFairShare::from_config(&FairShareConfig {
            default_weight: 1.0,
            default_output_tokens: 256,
            trust_output_token_estimate_header: trust_header,
            trust_request_model_header: false,
            tenant_weights: HashMap::new(),
            model_profiles: HashMap::new(),
        })
    }

    #[test]
    fn untrusted_client_estimate_cannot_reduce_reservation() {
        let mut headers = HeaderMap::new();
        headers.insert(OUTPUT_TOKEN_ESTIMATE_HEADER, "1".parse().unwrap());
        assert_eq!(output_token_estimate(&headers, &ledger(false)), 256);
    }

    #[test]
    fn trusted_proxy_estimate_is_honored() {
        let mut headers = HeaderMap::new();
        headers.insert(OUTPUT_TOKEN_ESTIMATE_HEADER, "128".parse().unwrap());
        assert_eq!(output_token_estimate(&headers, &ledger(true)), 128);
    }

    #[tokio::test]
    async fn local_adaptive_rejection_cancels_provisional_fair_share_charge() {
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
                default_output_tokens: 256,
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
        let tenant = TenantKey::new("header:alice");
        let scope_id = scheduler.fair_share_scope_for_test().unwrap();
        let AdmitOutcome::Admitted(mut permit) = scheduler
            .admit_for_tenant(
                Class::Default,
                RequestId("adaptive-rejected".into()),
                CancellationToken::new(),
                tenant.clone(),
                256,
            )
            .await
        else {
            panic!("fair-share request should be provisionally admitted");
        };
        assert_eq!(ledger.snapshot(scope_id, &tenant).1, 256);

        let mut response = Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .body(Body::empty())
            .unwrap();
        response.extensions_mut().insert(LocalAdaptiveRejection);
        assert!(consume_local_adaptive_rejection_marker(
            &mut response,
            &mut permit
        ));
        assert!(response
            .extensions()
            .get::<LocalAdaptiveRejection>()
            .is_none());
        assert_eq!(ledger.snapshot(scope_id, &tenant), (0, 0, 0, [0; 4]));

        drop(permit);
        assert_eq!(
            ledger.snapshot(scope_id, &tenant),
            (0, 0, 0, [0; 4]),
            "permit drop remains idempotent after cancellation"
        );
    }
}
