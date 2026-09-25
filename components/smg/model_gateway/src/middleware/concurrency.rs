//! Per-request concurrency limiting via a token bucket, with optional
//! queuing for backpressure.
//!
//! `ConcurrencyLimiter` wires a bounded `mpsc` channel that
//! `concurrency_limit_middleware` uses to enqueue requests when the
//! bucket is empty; `QueueProcessor` drains that channel and hands tokens
//! back to waiters. `TokenGuardBody` wraps the response body so the token
//! is only released after the entire stream has been delivered.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame};
use serde_json::json;
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};
use tracing::{debug, error, warn};

use super::token_bucket::TokenBucket;
use crate::{
    middleware::admission_metrics::{
        AdmissionActiveGuard, AdmissionPendingGuard, AdmissionQueuedGuard,
    },
    observability::metrics::{metrics_labels, Metrics},
    server::AppState,
};

/// A body wrapper that holds a token and returns it when the body is fully consumed or dropped.
/// This ensures that for streaming responses, the token is only returned after the entire
/// stream has been sent to the client.
pub struct TokenGuardBody {
    inner: Body,
    /// The token bucket to return tokens to. Uses Option so we can take() on drop.
    token_bucket: Option<Arc<TokenBucket>>,
    /// Number of tokens to return.
    tokens: f64,
    /// Active-admission gauge guard held through the full response body.
    active_guard: Option<AdmissionActiveGuard>,
    /// HTTP outcome returned by the protected route.
    status_code: StatusCode,
    /// True only after the wrapped body reaches end-of-stream.
    completed: bool,
}

impl TokenGuardBody {
    /// Create a new TokenGuardBody that will return tokens when dropped.
    pub fn new(inner: Body, token_bucket: Arc<TokenBucket>, tokens: f64) -> Self {
        let completed = inner.is_end_stream();
        Self {
            inner,
            token_bucket: Some(token_bucket),
            tokens,
            active_guard: None,
            status_code: StatusCode::OK,
            completed,
        }
    }

    fn new_with_admission(
        inner: Body,
        token_bucket: Option<Arc<TokenBucket>>,
        tokens: f64,
        active_guard: AdmissionActiveGuard,
        status_code: StatusCode,
    ) -> Self {
        let completed = inner.is_end_stream();
        Self {
            inner,
            token_bucket,
            tokens,
            active_guard: Some(active_guard),
            status_code,
            completed,
        }
    }
}

impl Drop for TokenGuardBody {
    fn drop(&mut self) {
        if let Some(active_guard) = &mut self.active_guard {
            if self.completed {
                active_guard.record_outcome(self.status_code.as_u16());
            } else {
                active_guard.record_interrupted();
            }
        }
        if let Some(bucket) = self.token_bucket.take() {
            debug!(
                "TokenGuardBody: stream ended, returning {} tokens to bucket",
                self.tokens
            );
            // Use lock-free sync return - no runtime needed, guaranteed token return
            bucket.return_tokens_sync(self.tokens);
        }
    }
}

impl http_body::Body for TokenGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // SAFETY: We never move the inner body, and Body is Unpin
        // (it's a type alias for UnsyncBoxBody which is Unpin)
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(cx);
        match &frame {
            Poll::Ready(None) => this.completed = true,
            Poll::Ready(Some(Err(_))) => this.completed = false,
            _ => {}
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<(), StatusCode>>,
    /// Admission queue slot held until the request leaves the queue.
    _queue_permit: OwnedSemaphorePermit,
}

/// Queue processor that handles queued requests
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        debug!("Starting concurrency queue processor");

        // Process requests in a single task to reduce overhead
        while let Some(queued) = self.queue_rx.recv().await {
            let QueuedRequest {
                queued_at,
                mut permit_tx,
                _queue_permit,
            } = queued;
            if permit_tx.is_closed() {
                debug!("Queue: requester disappeared before processing");
                continue;
            }

            // Check timeout immediately
            let elapsed = queued_at.elapsed();
            if elapsed >= self.queue_timeout {
                warn!("Request already timed out in queue");
                let _ = permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                continue;
            }

            let remaining_timeout = self.queue_timeout - elapsed;

            // Try to acquire token for this request
            if self.token_bucket.try_acquire(1.0).is_ok() {
                // Got token immediately
                debug!("Queue: acquired token immediately for queued request");
                if permit_tx.send(Ok(())).is_err() {
                    self.token_bucket.return_tokens_sync(1.0);
                }
            } else {
                // Need to wait for token
                let token_bucket = self.token_bucket.clone();

                // Spawn task only when we actually need to wait
                #[expect(
                    clippy::disallowed_methods,
                    reason = "fire-and-forget permit acquisition: task is bounded by remaining_timeout and communicates via oneshot; dropping the JoinHandle detaches the task but it self-terminates"
                )]
                tokio::spawn(async move {
                    let _queue_permit = _queue_permit;
                    tokio::select! {
                        _ = permit_tx.closed() => {
                            debug!("Queue: requester disappeared while waiting for token");
                        }
                        result = token_bucket.acquire_timeout(1.0, remaining_timeout) => {
                            if result.is_ok() {
                                debug!("Queue: acquired token after waiting");
                                if permit_tx.send(Ok(())).is_err() {
                                    token_bucket.return_tokens_sync(1.0);
                                }
                            } else {
                                warn!("Queue: request timed out waiting for token");
                                let _ = permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                            }
                        }
                    }
                });
            }
        }

        warn!("Concurrency queue processor shutting down");
    }
}

/// State for the concurrency limiter
pub struct ConcurrencyLimiter {
    pub queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    pub queue_slots: Option<Arc<Semaphore>>,
}

impl ConcurrencyLimiter {
    /// Create new concurrency limiter with optional queue
    pub fn new(
        token_bucket: Option<Arc<TokenBucket>>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        match (token_bucket, queue_size) {
            (None, _) => (
                Self {
                    queue_tx: None,
                    queue_slots: None,
                },
                None,
            ),
            (Some(bucket), size) if size > 0 => {
                let (queue_tx, queue_rx) = mpsc::channel(size);
                let processor = QueueProcessor::new(bucket, queue_rx, queue_timeout);
                (
                    Self {
                        queue_tx: Some(queue_tx),
                        queue_slots: Some(Arc::new(Semaphore::new(size))),
                    },
                    Some(processor),
                )
            }
            (Some(_), _) => (
                Self {
                    queue_tx: None,
                    queue_slots: None,
                },
                None,
            ),
        }
    }
}

async fn run_admitted_request(
    request: Request<Body>,
    next: Next,
    token_bucket: Option<Arc<TokenBucket>>,
    mut pending_guard: AdmissionPendingGuard,
) -> Response {
    pending_guard.resolve();
    Metrics::record_http_admission_admitted();
    let active_guard = AdmissionActiveGuard::new();
    let response = next.run(request).await;
    let status_code = response.status();
    let (parts, body) = response.into_parts();
    let guarded_body =
        TokenGuardBody::new_with_admission(body, token_bucket, 1.0, active_guard, status_code);
    Response::from_parts(parts, Body::new(guarded_body))
}

/// Enforce the optional cluster-wide requests-per-second ceiling before either
/// the legacy or priority-aware local admission path runs.
pub async fn global_rate_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(limit) = app_state
        .context
        .router_config
        .global_rate_limit_requests_per_second
    else {
        return next.run(request).await;
    };

    let Some(mesh_adapters) = &app_state.mesh_adapters else {
        error!(
            "Global rate limiting is configured at {} req/s but mesh is unavailable",
            limit
        );
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
        Metrics::record_http_admission_received();
        Metrics::record_http_admission_rejected();
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Global rate limiting requires mesh"
            })),
        )
            .into_response();
    };

    let (is_exceeded, current_count) = mesh_adapters
        .rate_limit()
        .check_and_increment("global", limit);
    if is_exceeded {
        debug!(
            "Global rate limit exceeded: {}/{} req/s",
            current_count, limit
        );
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
        Metrics::record_http_admission_received();
        Metrics::record_http_admission_rejected();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": "Rate limit exceeded",
                "current_count": current_count,
                "limit": limit
            })),
        )
            .into_response();
    }

    next.run(request).await
}

/// Middleware function for concurrency limiting with optional queuing
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    Metrics::record_http_admission_received();
    let mut pending_guard = AdmissionPendingGuard::new();

    let token_bucket = match &app_state.context.rate_limiter {
        Some(bucket) => bucket.clone(),
        None => {
            // Rate limiting disabled, pass through immediately
            return run_admitted_request(request, next, None, pending_guard).await;
        }
    };

    // Try to acquire token immediately
    if token_bucket.try_acquire(1.0).is_ok() {
        debug!("Acquired token immediately");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
        run_admitted_request(request, next, Some(token_bucket), pending_guard).await
    } else {
        // No tokens available, try to queue if enabled
        if let (Some(queue_tx), Some(queue_slots)) = (
            &app_state.concurrency_queue_tx,
            &app_state.concurrency_queue_slots,
        ) {
            debug!("No tokens available, attempting to queue request");

            // Create a channel for the token response
            let (permit_tx, permit_rx) = oneshot::channel();

            let queue_permit = match queue_slots.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    warn!("Request queue is full, returning 429");
                    Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                    Metrics::record_http_admission_rejected();
                    pending_guard.resolve();
                    return StatusCode::TOO_MANY_REQUESTS.into_response();
                }
            };
            let queued = QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
                _queue_permit: queue_permit,
            };

            // Try to send to queue
            match queue_tx.try_send(queued) {
                Ok(()) => {
                    let queued_guard = AdmissionQueuedGuard::new();
                    // Wait for token from queue processor
                    let permit_result = permit_rx.await;
                    drop(queued_guard);
                    match permit_result {
                        Ok(Ok(())) => {
                            debug!("Acquired token from queue");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
                            run_admitted_request(request, next, Some(token_bucket), pending_guard)
                                .await
                        }
                        Ok(Err(status)) => {
                            warn!("Queue returned error status: {}", status);
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            Metrics::record_http_admission_rejected();
                            pending_guard.resolve();
                            status.into_response()
                        }
                        Err(_) => {
                            error!("Queue response channel closed");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            Metrics::record_http_admission_rejected();
                            pending_guard.resolve();
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        }
                    }
                }
                Err(_) => {
                    warn!("Request queue is full, returning 429");
                    Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                    Metrics::record_http_admission_rejected();
                    pending_guard.resolve();
                    StatusCode::TOO_MANY_REQUESTS.into_response()
                }
            }
        } else {
            warn!("No tokens available and queuing is disabled, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_http_admission_rejected();
            pending_guard.resolve();
            StatusCode::TOO_MANY_REQUESTS.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_guard_body_preserves_exact_size_hint() {
        let inner = Body::from("measured response");
        let expected = inner.size_hint().exact();
        let guarded = TokenGuardBody::new(inner, Arc::new(TokenBucket::new(1, 0)), 1.0);

        assert_eq!(guarded.size_hint().exact(), expected);
    }

    #[test]
    fn configured_queue_capacity_bounds_all_outstanding_waiters() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (limiter, _processor) =
            ConcurrencyLimiter::new(Some(bucket), 2, Duration::from_secs(30));
        let slots = limiter.queue_slots.expect("queue slots");

        let first = slots.clone().try_acquire_owned().expect("first slot");
        let _second = slots.clone().try_acquire_owned().expect("second slot");
        assert!(slots.clone().try_acquire_owned().is_err());

        drop(first);
        assert!(slots.try_acquire_owned().is_ok());
    }

    #[test]
    fn disabled_queue_has_no_channel_or_slots() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (limiter, processor) =
            ConcurrencyLimiter::new(Some(bucket), 0, Duration::from_secs(30));

        assert!(limiter.queue_tx.is_none());
        assert!(limiter.queue_slots.is_none());
        assert!(processor.is_none());
    }

    #[tokio::test]
    async fn cancelled_waiter_releases_its_queue_slot_without_waiting_for_timeout() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        bucket.try_acquire(1.0).expect("exhaust initial token");
        let (limiter, processor) =
            ConcurrencyLimiter::new(Some(bucket), 1, Duration::from_secs(30));
        let queue_tx = limiter.queue_tx.expect("queue sender");
        let slots = limiter.queue_slots.expect("queue slots");
        let queue_permit = slots.clone().try_acquire_owned().expect("queue slot");
        let (permit_tx, permit_rx) = oneshot::channel();
        queue_tx
            .send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
                _queue_permit: queue_permit,
            })
            .await
            .expect("enqueue request");
        drop(permit_rx);

        let processor_task = tokio::spawn(processor.expect("processor").run());
        tokio::time::timeout(Duration::from_secs(1), async {
            while slots.available_permits() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled queue slot released");
        processor_task.abort();
    }
}
