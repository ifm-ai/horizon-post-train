//! Authenticated internal HTTP surface and inference-request bindings for
//! scheduler-backed capacity credits.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    extract::{Extension, Path},
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use smg_auth::RequestId;

use super::{
    capacity_credit::{
        ArmedCapacityCredit, CapacityCreditBinding, CapacityCreditError, CapacityCreditToken,
        HeldSchedulerPermit, IssuedCapacityCredit,
    },
    state::{SchedulerPartition, SchedulerState},
    Class,
};
use crate::{middleware::RouteRequestMeta, tenant::TenantKey};

pub const CAPACITY_CREDIT_HEADER: &str = "x-smg-capacity-credit";
pub const CAPACITY_CREDIT_GENERATION_HEADER: &str = "x-smg-capacity-credit-generation";
pub const CAPACITY_CREDIT_POLICY_EPOCH_HEADER: &str = "x-smg-fair-share-policy-epoch";
pub const CAPACITY_CREDIT_REQUEST_ID_HEADER: &str = "x-smg-capacity-credit-request-id";

static NEXT_CREDIT_ADMISSION_ID: AtomicU64 = AtomicU64::new(0);

fn next_credit_admission_id() -> RequestId {
    let id = NEXT_CREDIT_ADMISSION_ID.fetch_add(1, Ordering::Relaxed);
    RequestId(format!("credit-{id}"))
}

#[derive(Debug, Deserialize)]
pub(crate) struct IssueCapacityCreditRequest {
    pub generation: String,
    pub policy_epoch: u64,
    pub partition: String,
    pub model: String,
    pub request_id: String,
    pub estimated_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArmCapacityCreditRequest {
    #[serde(flatten)]
    pub binding: IssueCapacityCreditRequest,
    pub dispatch_id: String,
}

#[derive(Debug, Serialize)]
struct IssueCapacityCreditResponse {
    credit: String,
    generation: String,
    expires_in_ms: u64,
    newly_issued: bool,
}

#[derive(Debug, Serialize)]
struct ArmCapacityCreditResponse {
    generation: String,
    dispatch_id: String,
    expires_in_ms: u64,
    newly_armed: bool,
}

#[derive(Debug, Serialize)]
struct CapacityCreditCapabilitiesResponse {
    arm: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

pub(crate) struct PresentedCapacityCredit {
    pub token: CapacityCreditToken,
    pub binding: CapacityCreditBinding,
    pub partition: SchedulerPartition,
}

pub(crate) async fn issue_capacity_credit(
    Extension(state): Extension<Arc<SchedulerState>>,
    Extension(meta): Extension<RouteRequestMeta>,
    Json(request): Json<IssueCapacityCreditRequest>,
) -> Response {
    let Some(registry) = state.capacity_credits() else {
        return protocol_error(
            StatusCode::NOT_FOUND,
            "capacity_credits_disabled",
            "capacity credits are disabled",
            None,
        );
    };
    let Some(partition) = state.partition_exact(request.partition.trim()) else {
        return protocol_error(
            StatusCode::BAD_REQUEST,
            "unknown_partition",
            "capacity credit names an unknown admission partition",
            None,
        );
    };
    let canonical_model = state.canonical_model(request.model.trim());
    let binding = match CapacityCreditBinding::new(
        &request.generation,
        request.policy_epoch,
        partition.name.as_ref(),
        canonical_model.as_ref(),
        meta.tenant_key().clone(),
        &request.request_id,
        request.estimated_output_tokens,
    ) {
        Ok(binding) => binding,
        Err(error) => return capacity_credit_error_response(error, None),
    };
    let tenant = binding.tenant().clone();
    let estimated_output_tokens = binding.estimated_output_tokens();
    let profile = partition
        .scheduler
        .fair_share()
        .map_or(super::fair_share::FairShareProfile::Global, |ledger| {
            state.fair_share_profile_for_model(binding.model(), ledger)
        });
    let issued = registry.issue_with(binding, || {
        partition
            .scheduler
            .acquire_external_credit_for_tenant_profile(
                Class::Default,
                next_credit_admission_id(),
                tenant,
                profile,
                estimated_output_tokens,
            )
            .map(HeldSchedulerPermit::new)
            .ok_or(CapacityCreditError::NoCapacity)
    });
    match issued {
        Ok(issued) => issue_response(&request.generation, issued),
        Err(error) => capacity_credit_error_response(
            error,
            Some(partition.scheduler.retry_after_secs(Class::Default)),
        ),
    }
}

fn issue_response(generation: &str, issued: IssuedCapacityCredit) -> Response {
    let expires_in_ms = issued
        .expires_at
        .saturating_duration_since(Instant::now())
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    (
        StatusCode::OK,
        Json(IssueCapacityCreditResponse {
            credit: issued.token.expose_secret(),
            generation: generation.to_string(),
            expires_in_ms,
            newly_issued: issued.newly_issued,
        }),
    )
        .into_response()
}

pub(crate) async fn cancel_capacity_credit(
    Extension(state): Extension<Arc<SchedulerState>>,
    Path(encoded): Path<String>,
) -> Response {
    let Some(registry) = state.capacity_credits() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let token = match CapacityCreditToken::parse(&encoded) {
        Ok(token) => token,
        Err(error) => return capacity_credit_error_response(error, None),
    };
    registry.cancel(&token);
    StatusCode::NO_CONTENT.into_response()
}

pub(crate) async fn capacity_credit_capabilities() -> Response {
    (
        StatusCode::OK,
        Json(CapacityCreditCapabilitiesResponse { arm: "v1" }),
    )
        .into_response()
}

pub(crate) async fn arm_capacity_credit(
    Extension(state): Extension<Arc<SchedulerState>>,
    Extension(meta): Extension<RouteRequestMeta>,
    Path(encoded): Path<String>,
    Json(request): Json<ArmCapacityCreditRequest>,
) -> Response {
    let Some(registry) = state.capacity_credits() else {
        return protocol_error(
            StatusCode::NOT_FOUND,
            "capacity_credits_disabled",
            "capacity credits are disabled",
            None,
        );
    };
    let token = match CapacityCreditToken::parse(&encoded) {
        Ok(token) => token,
        Err(error) => return capacity_credit_error_response(error, None),
    };
    let binding_request = &request.binding;
    let Some(partition) = state.partition_exact(binding_request.partition.trim()) else {
        return protocol_error(
            StatusCode::BAD_REQUEST,
            "unknown_partition",
            "capacity credit names an unknown admission partition",
            None,
        );
    };
    let canonical_model = state.canonical_model(binding_request.model.trim());
    let binding = match CapacityCreditBinding::new(
        &binding_request.generation,
        binding_request.policy_epoch,
        partition.name.as_ref(),
        canonical_model.as_ref(),
        meta.tenant_key().clone(),
        &binding_request.request_id,
        binding_request.estimated_output_tokens,
    ) {
        Ok(binding) => binding,
        Err(error) => return capacity_credit_error_response(error, None),
    };
    match registry.arm(&token, &binding, &request.dispatch_id) {
        Ok(armed) => arm_response(&binding_request.generation, &request.dispatch_id, armed),
        Err(error) => capacity_credit_error_response(error, None),
    }
}

fn arm_response(generation: &str, dispatch_id: &str, armed: ArmedCapacityCredit) -> Response {
    let expires_in_ms = armed
        .expires_at
        .saturating_duration_since(Instant::now())
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    (
        StatusCode::OK,
        Json(ArmCapacityCreditResponse {
            generation: generation.to_string(),
            dispatch_id: dispatch_id.to_string(),
            expires_in_ms,
            newly_armed: armed.newly_armed,
        }),
    )
        .into_response()
}

pub(crate) fn presented_capacity_credit(
    headers: &HeaderMap,
    tenant: &TenantKey,
    state: &SchedulerState,
) -> Result<Option<PresentedCapacityCredit>, CapacityCreditError> {
    let Some(encoded_token) = optional_header(headers, CAPACITY_CREDIT_HEADER)? else {
        return Ok(None);
    };
    let token = CapacityCreditToken::parse(encoded_token)?;
    let generation = required_header(headers, CAPACITY_CREDIT_GENERATION_HEADER)?;
    let policy_epoch = required_header(headers, CAPACITY_CREDIT_POLICY_EPOCH_HEADER)?
        .parse::<u64>()
        .map_err(|_| CapacityCreditError::InvalidBinding("policy epoch"))?;
    let partition_name = required_header(headers, super::state::ADMISSION_PARTITION_HEADER)?;
    let partition = state
        .partition_exact(partition_name)
        .ok_or(CapacityCreditError::InvalidBinding("partition"))?;
    let raw_model = required_header(headers, super::fair_share::REQUEST_MODEL_HEADER)?;
    let canonical_model = state.canonical_model(raw_model);
    let request_id = required_header(headers, CAPACITY_CREDIT_REQUEST_ID_HEADER)?;
    let estimated_output_tokens =
        required_header(headers, super::fair_share::OUTPUT_TOKEN_ESTIMATE_HEADER)?
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or(CapacityCreditError::InvalidBinding(
                "estimated output tokens",
            ))?;
    let binding = CapacityCreditBinding::new(
        generation,
        policy_epoch,
        partition.name.as_ref(),
        canonical_model.as_ref(),
        tenant.clone(),
        request_id,
        estimated_output_tokens,
    )?;
    Ok(Some(PresentedCapacityCredit {
        token,
        binding,
        partition,
    }))
}

fn optional_header<'a>(
    headers: &'a HeaderMap,
    name: &'static str,
) -> Result<Option<&'a str>, CapacityCreditError> {
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .ok()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or(CapacityCreditError::InvalidBinding(
                    "capacity credit header",
                ))
        })
        .transpose()
}

fn required_header<'a>(
    headers: &'a HeaderMap,
    name: &'static str,
) -> Result<&'a str, CapacityCreditError> {
    optional_header(headers, name)?.ok_or(CapacityCreditError::InvalidBinding(name))
}

pub(crate) fn capacity_credit_required_response() -> Response {
    protocol_error(
        StatusCode::TOO_MANY_REQUESTS,
        "capacity_credit_required",
        "this endpoint requires a fair-share capacity credit; use the Comet client wrapper and retry",
        Some(1),
    )
}

pub(crate) fn capacity_credit_error_response(
    error: CapacityCreditError,
    retry_after: Option<u64>,
) -> Response {
    let (status, code) = match error {
        CapacityCreditError::InvalidToken | CapacityCreditError::InvalidBinding(_) => {
            (StatusCode::BAD_REQUEST, "invalid_capacity_credit")
        }
        CapacityCreditError::NoCapacity => (StatusCode::TOO_MANY_REQUESTS, "capacity_unavailable"),
        CapacityCreditError::WrongGeneration => {
            (StatusCode::CONFLICT, "capacity_credit_wrong_generation")
        }
        CapacityCreditError::RequestConflict => {
            (StatusCode::CONFLICT, "capacity_credit_request_conflict")
        }
        CapacityCreditError::Unknown => (StatusCode::CONFLICT, "capacity_credit_unknown"),
        CapacityCreditError::BindingMismatch => {
            (StatusCode::CONFLICT, "capacity_credit_binding_mismatch")
        }
        CapacityCreditError::Expired => (StatusCode::CONFLICT, "capacity_credit_expired"),
        CapacityCreditError::AlreadyRedeemed => {
            (StatusCode::CONFLICT, "capacity_credit_already_redeemed")
        }
        CapacityCreditError::AlreadyArmed => {
            (StatusCode::CONFLICT, "capacity_credit_already_armed")
        }
        CapacityCreditError::Cancelled => (StatusCode::CONFLICT, "capacity_credit_cancelled"),
        CapacityCreditError::InternalState => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "capacity_credit_internal",
        ),
    };
    protocol_error(status, code, &error.to_string(), retry_after)
}

fn protocol_error(
    status: StatusCode,
    code: &'static str,
    message: &str,
    retry_after: Option<u64>,
) -> Response {
    let mut response = (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                code,
                message: message.to_string(),
            },
        }),
    )
        .into_response();
    if let Some(seconds) = retry_after.filter(|seconds| *seconds > 0) {
        if let Ok(value) = HeaderValue::from_str(&seconds.to_string()) {
            response.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        sync::atomic::{AtomicU16, Ordering},
        time::Duration,
    };

    use axum::{
        body::Body,
        http::{header::AUTHORIZATION, Request},
        middleware::from_fn_with_state,
        routing::{get, post},
        Router,
    };
    use http_body_util::BodyExt;
    use tempfile::NamedTempFile;
    use tokio::sync::watch;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::RouterConfig,
        middleware::{
            auth_middleware,
            scheduler::{state::AdaptiveCapacityProvider, AdmissionMode},
            AuthConfig, TokenBucket,
        },
        tenant::RouteRequestMeta,
        worker::{BasicWorkerBuilder, WorkerRegistry},
    };

    struct TestCapacityProvider {
        capacity: AtomicU16,
        revision: watch::Sender<u64>,
    }

    impl TestCapacityProvider {
        fn new(capacity: u16) -> Self {
            let (revision, _) = watch::channel(0);
            Self {
                capacity: AtomicU16::new(capacity),
                revision,
            }
        }

        fn set_capacity(&self, capacity: u16) {
            if self.capacity.swap(capacity, Ordering::SeqCst) != capacity {
                self.revision
                    .send_modify(|revision| *revision = revision.wrapping_add(1));
            }
        }
    }

    impl AdaptiveCapacityProvider for TestCapacityProvider {
        fn effective_capacity(&self, _partition: &str, static_capacity: u16) -> u16 {
            self.capacity.load(Ordering::SeqCst).min(static_capacity)
        }

        fn subscribe_capacity_changes(&self) -> watch::Receiver<u64> {
            self.revision.subscribe()
        }
    }

    fn request(request_id: &str) -> IssueCapacityCreditRequest {
        IssueCapacityCreditRequest {
            generation: "green-1".to_string(),
            policy_epoch: 7,
            partition: "kimi-k3".to_string(),
            model: "kimi-k3".to_string(),
            request_id: request_id.to_string(),
            estimated_output_tokens: 1_024,
        }
    }

    fn arm_request(request_id: &str, dispatch_id: &str) -> ArmCapacityCreditRequest {
        ArmCapacityCreditRequest {
            binding: request(request_id),
            dispatch_id: dispatch_id.to_string(),
        }
    }

    fn scheduler_state_with_required(
        capacity_credit_required: bool,
    ) -> (Arc<SchedulerState>, NamedTempFile) {
        scheduler_state_with_provider(capacity_credit_required, None)
    }

    fn scheduler_state_with_provider(
        capacity_credit_required: bool,
        capacity_provider: Option<Arc<dyn AdaptiveCapacityProvider>>,
    ) -> (Arc<SchedulerState>, NamedTempFile) {
        let priority_scheduler_adaptive_capacity = capacity_provider.is_some();
        let mut yaml = NamedTempFile::new().unwrap();
        write!(
            yaml,
            r#"
classes:
  system:
    reserved_floor: 0
    reserved_per_slot: 0
    queue_size: 1
    queue_timeout_secs: 60
    starvation_threshold_secs: 60
    can_preempt: false
  interactive:
    reserved_floor: 0
    reserved_per_slot: 0
    queue_size: 1
    queue_timeout_secs: 60
    starvation_threshold_secs: 60
    can_preempt: false
fair_share:
  default_output_tokens: 1024
  trust_output_token_estimate_header: true
  trust_request_model_header: true
  model_profiles:
    kimi-k3:
      tenant_weights:
        "header:junu": 100
      other_weight: 1
admission_partitions:
  kimi-k3:
    max_concurrent_requests: 1
    queue_size: 1
default_admission_partition: kimi-k3
"#
        )
        .unwrap();
        let mut config = RouterConfig {
            api_key: Some("service-key".to_string()),
            max_concurrent_requests: 1,
            queue_size: 1,
            priority_scheduler_enabled: true,
            priority_scheduler_config: Some(yaml.path().to_string_lossy().into_owned()),
            capacity_credit_generation: Some("green-1".to_string()),
            capacity_credit_required,
            priority_scheduler_adaptive_capacity,
            ..RouterConfig::default()
        };
        config.tenant_resolution.trust_tenant_header = true;
        config.tenant_resolution.prefer_trusted_tenant_header = true;
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://kimi-worker:50051")
                    .model(openai_protocol::model_card::ModelCard::new("kimi-k3"))
                    .label("max_running_requests", "1")
                    .status(openai_protocol::worker::WorkerStatus::Ready)
                    .build(),
            ))
            .unwrap();
        let mode = AdmissionMode::try_build_priority(&config, registry, None, capacity_provider)
            .unwrap_or_else(|error| panic!("capacity-credit scheduler should build: {error}"));
        let AdmissionMode::Priority(state) = mode else {
            panic!("capacity-credit scheduler should build");
        };
        (state, yaml)
    }

    fn scheduler_state() -> (Arc<SchedulerState>, NamedTempFile) {
        scheduler_state_with_required(false)
    }

    fn inference_headers(token: &str, request_id: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in [
            (CAPACITY_CREDIT_HEADER, token),
            (CAPACITY_CREDIT_GENERATION_HEADER, "green-1"),
            (CAPACITY_CREDIT_POLICY_EPOCH_HEADER, "7"),
            (super::super::ADMISSION_PARTITION_HEADER, "kimi-k3"),
            (super::super::REQUEST_MODEL_HEADER, "kimi-k3"),
            (CAPACITY_CREDIT_REQUEST_ID_HEADER, request_id),
            (super::super::OUTPUT_TOKEN_ESTIMATE_HEADER, "1024"),
        ] {
            headers.insert(name, HeaderValue::from_str(value).unwrap());
        }
        headers
    }

    fn inference_app(state: Arc<SchedulerState>, tenant: TenantKey) -> Router {
        Router::new()
            .route(
                "/",
                post(|| async { (StatusCode::OK, r#"{"usage":{"completion_tokens":37}}"#) }),
            )
            .route_layer(from_fn_with_state(
                state,
                super::super::priority_admission_middleware,
            ))
            .layer(Extension(RouteRequestMeta::new(tenant)))
    }

    fn proof_inspection_app(state: Arc<SchedulerState>, tenant: TenantKey) -> Router {
        Router::new()
            .route(
                "/",
                post(|Extension(meta): Extension<RouteRequestMeta>| async move {
                    let Some(proof) = meta
                        .extension::<super::super::SchedulerAdmissionProof>()
                        .cloned()
                    else {
                        return StatusCode::INTERNAL_SERVER_ERROR;
                    };
                    if proof
                        .try_claim(
                            meta.tenant_key(),
                            meta.request_charge_id(),
                            "kimi-k3",
                            "kimi-k3",
                        )
                        .is_err()
                    {
                        return StatusCode::FORBIDDEN;
                    }
                    if !matches!(
                        proof.try_claim(
                            meta.tenant_key(),
                            meta.request_charge_id(),
                            "kimi-k3",
                            "kimi-k3",
                        ),
                        Err(super::super::ProofError::AlreadyClaimed)
                    ) {
                        return StatusCode::CONFLICT;
                    }
                    StatusCode::OK
                }),
            )
            .route_layer(from_fn_with_state(
                state,
                super::super::priority_admission_middleware,
            ))
            .layer(Extension(RouteRequestMeta::new(tenant)))
    }

    fn adaptive_rejection_app(state: Arc<SchedulerState>, tenant: TenantKey) -> Router {
        Router::new()
            .route(
                "/",
                post(|| async {
                    let mut response = Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .body(Body::empty())
                        .unwrap();
                    response
                        .extensions_mut()
                        .insert(super::super::LocalAdaptiveRejection);
                    response
                }),
            )
            .route_layer(from_fn_with_state(
                state,
                super::super::priority_admission_middleware,
            ))
            .layer(Extension(RouteRequestMeta::new(tenant)))
    }

    async fn issue_token(
        state: &Arc<SchedulerState>,
        tenant: &TenantKey,
        request_id: &str,
    ) -> String {
        let issued = issue_capacity_credit(
            Extension(Arc::clone(state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Json(request(request_id)),
        )
        .await;
        assert_eq!(issued.status(), StatusCode::OK);
        response_json(issued).await["credit"]
            .as_str()
            .unwrap()
            .to_string()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn capabilities_are_exact_and_require_the_service_key() {
        let app = Router::new()
            .route(
                "/internal/capacity-credits/capabilities",
                get(capacity_credit_capabilities),
            )
            .route_layer(from_fn_with_state(
                AuthConfig::new(Some("service-key".to_string())),
                auth_middleware,
            ));

        let unauthorized = app
            .clone()
            .oneshot(
                Request::get("/internal/capacity-credits/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = app
            .oneshot(
                Request::get("/internal/capacity-credits/capabilities")
                    .header(AUTHORIZATION, "Bearer service-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        assert_eq!(
            response_json(authorized).await,
            serde_json::json!({ "arm": "v1" })
        );
    }

    #[tokio::test]
    async fn issue_is_idempotent_and_cancel_releases_capacity() {
        let (state, _yaml) = scheduler_state();
        let tenant = RouteRequestMeta::new(TenantKey::new("header:junu"));
        let first = issue_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(tenant.clone()),
            Json(request("request-1")),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        let first_json = response_json(first).await;
        let token = first_json["credit"].as_str().unwrap().to_string();
        assert_eq!(first_json["newly_issued"], true);

        let duplicate = issue_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(tenant.clone()),
            Json(request("request-1")),
        )
        .await;
        assert_eq!(duplicate.status(), StatusCode::OK);
        let duplicate_json = response_json(duplicate).await;
        assert_eq!(duplicate_json["credit"], token);
        assert_eq!(duplicate_json["newly_issued"], false);

        let full = issue_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(tenant.clone()),
            Json(request("request-2")),
        )
        .await;
        assert_eq!(full.status(), StatusCode::TOO_MANY_REQUESTS);

        let cancelled = cancel_capacity_credit(Extension(Arc::clone(&state)), Path(token)).await;
        assert_eq!(cancelled.status(), StatusCode::NO_CONTENT);
        let next = issue_capacity_credit(
            Extension(state),
            Extension(tenant),
            Json(request("request-2")),
        )
        .await;
        assert_eq!(next.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn arm_is_binding_checked_idempotent_and_keeps_the_credit_redeemable() {
        let (state, _yaml) = scheduler_state();
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-arm").await;

        let mut mismatch = arm_request("request-arm", "dispatch-1");
        mismatch.binding.estimated_output_tokens += 1;
        let mismatch_response = arm_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Path(token.clone()),
            Json(mismatch),
        )
        .await;
        assert_eq!(mismatch_response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(mismatch_response).await["error"]["code"],
            "capacity_credit_binding_mismatch"
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 1);

        let first = arm_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Path(token.clone()),
            Json(arm_request("request-arm", "dispatch-1")),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        let first_json = response_json(first).await;
        assert_eq!(first_json["generation"], "green-1");
        assert_eq!(first_json["dispatch_id"], "dispatch-1");
        assert_eq!(first_json["newly_armed"], true);
        assert!(first_json["expires_in_ms"].as_u64().unwrap() > 0);

        let repeated = arm_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Path(token.clone()),
            Json(arm_request("request-arm", "dispatch-1")),
        )
        .await;
        assert_eq!(repeated.status(), StatusCode::OK);
        let repeated_json = response_json(repeated).await;
        assert_eq!(repeated_json["newly_armed"], false);
        assert!(
            repeated_json["expires_in_ms"].as_u64().unwrap()
                <= first_json["expires_in_ms"].as_u64().unwrap()
        );

        let competing = arm_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Path(token.clone()),
            Json(arm_request("request-arm", "dispatch-2")),
        )
        .await;
        assert_eq!(competing.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(competing).await["error"]["code"],
            "capacity_credit_already_armed"
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 1);

        let headers = inference_headers(&token, "request-arm");
        let presented = presented_capacity_credit(&headers, &tenant, &state)
            .unwrap()
            .unwrap();
        let redeemed = state
            .capacity_credits()
            .unwrap()
            .redeem(&presented.token, &presented.binding)
            .unwrap();
        let permit = redeemed.payload.into_permit();
        assert!(permit.has_fair_share_reservation());

        let replay = arm_capacity_credit(
            Extension(state),
            Extension(RouteRequestMeta::new(tenant)),
            Path(token),
            Json(arm_request("request-arm", "dispatch-1")),
        )
        .await;
        assert_eq!(replay.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(replay).await["error"]["code"],
            "capacity_credit_already_redeemed"
        );
    }

    #[tokio::test]
    async fn arm_rejects_wrong_generation_without_consuming_the_credit() {
        let (state, _yaml) = scheduler_state();
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-arm").await;
        let mut wrong_generation = arm_request("request-arm", "dispatch-1");
        wrong_generation.binding.generation = "blue-1".to_string();

        let response = arm_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Path(token.clone()),
            Json(wrong_generation),
        )
        .await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(response).await["error"]["code"],
            "capacity_credit_wrong_generation"
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 1);

        let response = arm_capacity_credit(
            Extension(state),
            Extension(RouteRequestMeta::new(tenant)),
            Path(token),
            Json(arm_request("request-arm", "dispatch-1")),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn exact_binding_redeems_once_without_queueing() {
        let (state, _yaml) = scheduler_state();
        let tenant = TenantKey::new("header:junu");
        let issued = issue_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(RouteRequestMeta::new(tenant.clone())),
            Json(request("request-1")),
        )
        .await;
        let issued_json = response_json(issued).await;
        let headers = inference_headers(issued_json["credit"].as_str().unwrap(), "request-1");
        let presented = presented_capacity_credit(&headers, &tenant, &state)
            .unwrap()
            .unwrap();
        let mut redeemed = state
            .capacity_credits()
            .unwrap()
            .redeem(&presented.token, &presented.binding)
            .unwrap();
        assert!(redeemed
            .payload
            .bind_redeemed_capacity_credit(&redeemed.binding, uuid::Uuid::now_v7()));
        let permit = redeemed.payload.into_permit();
        assert!(permit.has_fair_share_reservation());
        assert!(permit.fair_share_admission_proof().is_some());
        assert!(matches!(
            state
                .capacity_credits()
                .unwrap()
                .redeem(&presented.token, &presented.binding),
            Err(CapacityCreditError::AlreadyRedeemed)
        ));
    }

    #[tokio::test]
    async fn redeemed_credit_inserts_exact_one_shot_proof_into_tenant_metadata() {
        let (state, _yaml) = scheduler_state();
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-proof").await;
        let headers = inference_headers(&token, "request-proof");
        let mut request = Request::post("/").body(Body::empty()).unwrap();
        *request.headers_mut() = headers;

        let response = proof_inspection_app(state, tenant)
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn required_response_is_retryable_and_explicit() {
        let response = capacity_credit_required_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[RETRY_AFTER], "1");
    }

    #[tokio::test]
    async fn middleware_requires_credit_when_enabled() {
        let (state, _yaml) = scheduler_state_with_required(true);
        let tenant = TenantKey::new("header:junu");
        let response = inference_app(state, tenant)
            .oneshot(Request::post("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[RETRY_AFTER], "1");
        let json = response_json(response).await;
        assert_eq!(json["error"]["code"], "capacity_credit_required");
    }

    #[tokio::test]
    async fn middleware_redeems_credit_and_settles_terminal_usage() {
        let (state, _yaml) = scheduler_state_with_required(true);
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-1").await;
        let partition = state.partition_exact("kimi-k3").unwrap();
        let ledger = partition.scheduler.fair_share().unwrap();
        let scope_id = partition.scheduler.fair_share_scope_for_test().unwrap();

        let mut inference_request = Request::post("/").body(Body::empty()).unwrap();
        *inference_request.headers_mut() = inference_headers(&token, "request-1");
        let response = inference_app(Arc::clone(&state), tenant.clone())
            .oneshot(inference_request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["usage"]["completion_tokens"], 37);
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (37, 0, 0, [0; 4])
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 0);
    }

    #[tokio::test]
    async fn binding_mismatch_does_not_consume_credit() {
        let (state, _yaml) = scheduler_state_with_required(true);
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-1").await;
        let app = inference_app(Arc::clone(&state), tenant);

        let mut wrong = Request::post("/").body(Body::empty()).unwrap();
        *wrong.headers_mut() = inference_headers(&token, "different-request");
        let wrong_response = app.clone().oneshot(wrong).await.unwrap();
        assert_eq!(wrong_response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(wrong_response).await["error"]["code"],
            "capacity_credit_binding_mismatch"
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 1);

        let mut correct = Request::post("/").body(Body::empty()).unwrap();
        *correct.headers_mut() = inference_headers(&token, "request-1");
        let correct_response = app.oneshot(correct).await.unwrap();
        assert_eq!(correct_response.status(), StatusCode::OK);
        response_json(correct_response).await;
        assert_eq!(state.capacity_credits().unwrap().active_len(), 0);
    }

    #[tokio::test]
    async fn local_adaptive_rejection_releases_credit_without_token_charge() {
        let (state, _yaml) = scheduler_state_with_required(true);
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-1").await;
        let partition = state.partition_exact("kimi-k3").unwrap();
        let ledger = partition.scheduler.fair_share().unwrap();
        let scope_id = partition.scheduler.fair_share_scope_for_test().unwrap();
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (0, 1_024, 1, [0; 4])
        );

        let mut request = Request::post("/").body(Body::empty()).unwrap();
        *request.headers_mut() = inference_headers(&token, "request-1");
        let response = adaptive_rejection_app(Arc::clone(&state), tenant.clone())
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response
            .extensions()
            .get::<super::super::LocalAdaptiveRejection>()
            .is_none());
        response.into_body().collect().await.unwrap();
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (0, 0, 0, [0; 4])
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 0);
    }

    #[tokio::test]
    async fn local_rps_rejection_releases_credit_without_token_charge() {
        let (mut state, _yaml) = scheduler_state_with_required(true);
        let bucket = Arc::new(TokenBucket::new(1, 0));
        bucket
            .try_acquire(1.0)
            .expect("test should exhaust the single RPS token");
        Arc::get_mut(&mut state)
            .expect("test owns the only scheduler-state reference")
            .rate_limiter = Some(bucket);
        let tenant = TenantKey::new("header:junu");
        let token = issue_token(&state, &tenant, "request-1").await;
        let partition = state.partition_exact("kimi-k3").unwrap();
        let ledger = partition.scheduler.fair_share().unwrap();
        let scope_id = partition.scheduler.fair_share_scope_for_test().unwrap();

        let mut inference_request = Request::post("/").body(Body::empty()).unwrap();
        *inference_request.headers_mut() = inference_headers(&token, "request-1");
        let response = inference_app(Arc::clone(&state), tenant.clone())
            .oneshot(inference_request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (0, 0, 0, [0; 4])
        );
        assert_eq!(state.capacity_credits().unwrap().active_len(), 0);

        let replacement = issue_capacity_credit(
            Extension(state),
            Extension(RouteRequestMeta::new(tenant)),
            Json(request("request-2")),
        )
        .await;
        assert_eq!(replacement.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn adaptive_zero_capacity_issues_nothing_then_real_growth_issues_exactly_one() {
        let provider = Arc::new(TestCapacityProvider::new(0));
        let (state, _yaml) = scheduler_state_with_provider(
            true,
            Some(Arc::clone(&provider) as Arc<dyn AdaptiveCapacityProvider>),
        );
        let tenant = RouteRequestMeta::new(TenantKey::new("header:junu"));
        let unavailable = issue_capacity_credit(
            Extension(Arc::clone(&state)),
            Extension(tenant.clone()),
            Json(request("request-1")),
        )
        .await;
        assert_eq!(unavailable.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(state.capacity_credits().unwrap().active_len(), 0);

        provider.set_capacity(1);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let response = issue_capacity_credit(
                Extension(Arc::clone(&state)),
                Extension(tenant.clone()),
                Json(request("request-1")),
            )
            .await;
            if response.status() == StatusCode::OK {
                break;
            }
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(state.capacity_credits().unwrap().active_len(), 1);

        let second = issue_capacity_credit(
            Extension(state),
            Extension(tenant),
            Json(request("request-2")),
        )
        .await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
