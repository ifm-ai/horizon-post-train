//! HTTP mapping for priority-scheduler admission outcomes.
//!
//! [`SchedulerError`] is the response-facing form of a rejected admission
//! (and of a mid-flight preemption). The admission middleware converts a
//! [`RejectionReason`] into one of these; long-running handlers return
//! [`SchedulerError::Preempted`] when their cancel token fires.

use axum::{
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};

use super::engine::RejectionReason;
use crate::routers::error::create_error;

/// Response header set on a preempted request so clients/proxies can tell a
/// preemption 503 apart from an ordinary overload 503.
pub const HEADER_X_SMG_PREEMPTED: &str = "X-SMG-Preempted";

/// Client-Closed-Request (nginx convention). Used for `ClientCancelled`,
/// which is essentially never read — the client has already disconnected.
const STATUS_CLIENT_CLOSED_REQUEST: u16 = 499;

/// How the scheduler's admission decision surfaces to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    /// Shared global queue budget exhausted. → 429 + Retry-After.
    QueueFull,
    /// Queued waiter aged past `queue_timeout`. → 408 + Retry-After.
    QueueTimeout,
    /// Cancelled in-flight to admit a higher-priority waiter, before TTFT.
    /// → 503 + `Retry-After: 1` + `X-SMG-Preempted: true`.
    Preempted,
    /// The client disconnected before admission completed. → 499 (never
    /// actually read; the connection is gone).
    ClientCancelled,
}

impl SchedulerError {
    fn status(self) -> StatusCode {
        match self {
            Self::QueueFull => StatusCode::TOO_MANY_REQUESTS,
            Self::QueueTimeout => StatusCode::REQUEST_TIMEOUT,
            Self::Preempted => StatusCode::SERVICE_UNAVAILABLE,
            // `unwrap_or` (not `unwrap`/`expect`, both denied) — 499 is a
            // valid code so the fallback is never taken.
            Self::ClientCancelled => StatusCode::from_u16(STATUS_CLIENT_CLOSED_REQUEST)
                .unwrap_or(StatusCode::REQUEST_TIMEOUT),
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::QueueFull => "scheduler_queue_full",
            Self::QueueTimeout => "scheduler_queue_timeout",
            Self::Preempted => "scheduler_preempted",
            Self::ClientCancelled => "scheduler_client_cancelled",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::QueueFull => "global request queue is full",
            Self::QueueTimeout => "timed out waiting for an admission slot",
            Self::Preempted => "request preempted by higher-priority traffic",
            Self::ClientCancelled => "client closed the request before admission",
        }
    }

    /// Render a scheduler saturation response with the router's current
    /// quantitative retry estimate. Preemption keeps its fixed one-second
    /// retry hint; queue-full and queue-timeout responses use `seconds`.
    pub fn into_response_with_retry_after(self, seconds: u64) -> Response {
        self.response(Some(seconds.max(1)))
    }

    fn response(self, saturation_retry_after: Option<u64>) -> Response {
        let mut resp = create_error(self.status(), self.code(), self.message());
        let retry_after = match self {
            Self::Preempted => Some(1),
            Self::QueueFull | Self::QueueTimeout => saturation_retry_after,
            Self::ClientCancelled => None,
        };
        if let Some(seconds) = retry_after {
            if let Ok(value) = HeaderValue::from_str(&seconds.to_string()) {
                resp.headers_mut().insert(RETRY_AFTER, value);
            }
        }
        if self == Self::Preempted {
            resp.headers_mut()
                .insert(HEADER_X_SMG_PREEMPTED, HeaderValue::from_static("true"));
        }
        resp
    }
}

impl From<RejectionReason> for SchedulerError {
    fn from(reason: RejectionReason) -> Self {
        match reason {
            RejectionReason::QueueFull => Self::QueueFull,
            RejectionReason::QueueTimeout => Self::QueueTimeout,
            RejectionReason::Preempted => Self::Preempted,
            RejectionReason::ClientCancelled => Self::ClientCancelled,
        }
    }
}

impl IntoResponse for SchedulerError {
    fn into_response(self) -> Response {
        // Reuse the gateway's standard error shape (JSON body + the
        // X-SMG-Error-Code header), then layer retry/preemption headers on
        // top. Callers that know the live saturation estimate use
        // `into_response_with_retry_after` instead.
        self.response(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_full_maps_to_429() {
        let resp = SchedulerError::QueueFull.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_queue_timeout_maps_to_408() {
        let resp = SchedulerError::QueueTimeout.into_response();
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn test_preempted_maps_to_503_with_headers() {
        let resp = SchedulerError::Preempted.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(RETRY_AFTER).map(|v| v.to_str().unwrap()),
            Some("1")
        );
        assert_eq!(
            resp.headers()
                .get(HEADER_X_SMG_PREEMPTED)
                .map(|v| v.to_str().unwrap()),
            Some("true")
        );
    }

    #[test]
    fn test_client_cancelled_maps_to_499() {
        let resp = SchedulerError::ClientCancelled.into_response();
        assert_eq!(resp.status().as_u16(), 499);
    }

    #[test]
    fn test_non_preempt_has_no_preempt_headers() {
        let resp = SchedulerError::QueueFull.into_response();
        assert!(resp.headers().get(HEADER_X_SMG_PREEMPTED).is_none());
        assert!(resp.headers().get(RETRY_AFTER).is_none());
    }

    #[test]
    fn test_queue_full_with_estimate_sets_retry_after() {
        let resp = SchedulerError::QueueFull.into_response_with_retry_after(23);
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers().get(RETRY_AFTER).unwrap(), "23");
        assert!(resp.headers().get(HEADER_X_SMG_PREEMPTED).is_none());
    }

    #[test]
    fn test_queue_timeout_with_estimate_sets_retry_after() {
        let resp = SchedulerError::QueueTimeout.into_response_with_retry_after(7);
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(resp.headers().get(RETRY_AFTER).unwrap(), "7");
    }

    #[test]
    fn test_from_rejection_reason_is_one_to_one() {
        assert_eq!(
            SchedulerError::from(RejectionReason::QueueFull),
            SchedulerError::QueueFull
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::QueueTimeout),
            SchedulerError::QueueTimeout
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::Preempted),
            SchedulerError::Preempted
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::ClientCancelled),
            SchedulerError::ClientCancelled
        );
    }
}
