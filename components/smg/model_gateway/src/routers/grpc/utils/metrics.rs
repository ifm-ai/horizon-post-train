//! Metrics helper functions (shared by HTTP routers and gRPC pipeline).

use http::StatusCode;

use crate::observability::metrics::metrics_labels;

/// Map route path to endpoint label for metrics
pub(crate) fn route_to_endpoint(route: &str) -> &'static str {
    match route {
        "/v1/chat/completions" => metrics_labels::ENDPOINT_CHAT,
        "/generate" => metrics_labels::ENDPOINT_GENERATE,
        "/v1/completions" => metrics_labels::ENDPOINT_COMPLETIONS,
        "/v1/rerank" => metrics_labels::ENDPOINT_RERANK,
        "/v1/responses" => metrics_labels::ENDPOINT_RESPONSES,
        "/v1/messages" => metrics_labels::ENDPOINT_MESSAGES,
        "/v1/audio/transcriptions" => metrics_labels::ENDPOINT_AUDIO_TRANSCRIPTIONS,
        _ => "other",
    }
}

/// Map HTTP status code to error type label for metrics
pub(crate) fn error_type_from_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => metrics_labels::ERROR_VALIDATION,
        404 => metrics_labels::ERROR_NO_WORKERS,
        408 | 504 => metrics_labels::ERROR_TIMEOUT,
        429 => metrics_labels::ERROR_ADMISSION_REJECTED,
        500..=599 => metrics_labels::ERROR_BACKEND,
        _ => metrics_labels::ERROR_INTERNAL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_admission_rejection_is_not_an_internal_error() {
        assert_eq!(
            error_type_from_status(StatusCode::TOO_MANY_REQUESTS),
            metrics_labels::ERROR_ADMISSION_REJECTED
        );
    }

    #[test]
    fn representative_router_error_mappings_remain_stable() {
        for (status, expected) in [
            (StatusCode::BAD_REQUEST, metrics_labels::ERROR_VALIDATION),
            (StatusCode::NOT_FOUND, metrics_labels::ERROR_NO_WORKERS),
            (StatusCode::REQUEST_TIMEOUT, metrics_labels::ERROR_TIMEOUT),
            (StatusCode::GATEWAY_TIMEOUT, metrics_labels::ERROR_TIMEOUT),
            (StatusCode::BAD_GATEWAY, metrics_labels::ERROR_BACKEND),
            (StatusCode::FORBIDDEN, metrics_labels::ERROR_INTERNAL),
        ] {
            assert_eq!(error_type_from_status(status), expected);
        }
    }
}
