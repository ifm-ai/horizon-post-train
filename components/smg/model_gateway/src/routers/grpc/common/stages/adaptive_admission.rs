//! Adaptive predicted-work admission after request tokenization.

use async_trait::async_trait;
use axum::{
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    response::Response,
};

use super::PipelineStage;
use crate::{
    middleware::scheduler::{
        LocalAdaptiveRejection, SchedulerAdmissionProof, ADMISSION_PARTITION_HEADER,
    },
    routers::{
        error,
        grpc::{
            adaptive_admission::{
                PredictionFeatures, FLAG_MULTIPLE_COMPLETIONS, FLAG_REASONING, FLAG_STREAMING,
                FLAG_STRUCTURED_OUTPUT, FLAG_TOOLS,
            },
            context::{
                PendingDistributionSeed, RequestContext, RequestType,
                StreamingDistributionSeedScope,
            },
        },
    },
};

const COMET_USER_HEADER: &str = "x-comet-user";
const COMET_WORKLOAD_TYPE_HEADER: &str = "x-comet-workload-type";

pub(crate) struct AdaptiveAdmissionStage;

#[derive(Clone, Copy)]
struct GenerationShape {
    endpoint: &'static str,
    max_output_tokens: Option<u32>,
    flags: u16,
}

impl GenerationShape {
    fn for_request(request: &RequestType) -> Option<Self> {
        match request {
            RequestType::Chat(request) => {
                #[expect(
                    deprecated,
                    reason = "max_tokens remains an OpenAI compatibility fallback"
                )]
                let per_completion_limit = request.max_completion_tokens.or(request.max_tokens);
                let multiplicity = request.n.unwrap_or(1).max(1);
                let mut flags = 0;
                if multiplicity > 1 {
                    flags |= FLAG_MULTIPLE_COMPLETIONS;
                }
                if request
                    .tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty())
                {
                    flags |= FLAG_TOOLS;
                }
                if request.response_format.is_some()
                    || request.regex.is_some()
                    || request.ebnf.is_some()
                {
                    flags |= FLAG_STRUCTURED_OUTPUT;
                }
                if request.reasoning_effort.is_some() {
                    flags |= FLAG_REASONING;
                }
                if request.stream {
                    flags |= FLAG_STREAMING;
                }
                Some(Self {
                    endpoint: "chat",
                    max_output_tokens: multiplied_limit(per_completion_limit, multiplicity),
                    flags,
                })
            }
            RequestType::Generate(request) => {
                let params = request.sampling_params.as_ref();
                let multiplicity = params.and_then(|p| p.n).unwrap_or(1).max(1);
                let mut flags = 0;
                if multiplicity > 1 {
                    flags |= FLAG_MULTIPLE_COMPLETIONS;
                }
                if params.is_some_and(|p| {
                    p.json_schema.is_some() || p.regex.is_some() || p.ebnf.is_some()
                }) {
                    flags |= FLAG_STRUCTURED_OUTPUT;
                }
                if request.stream {
                    flags |= FLAG_STREAMING;
                }
                Some(Self {
                    endpoint: "generate",
                    max_output_tokens: multiplied_limit(
                        params.and_then(|p| p.max_new_tokens),
                        multiplicity,
                    ),
                    flags,
                })
            }
            RequestType::Completion(request) => {
                let returned = request.n.unwrap_or(1).max(1);
                let generated = request.best_of.unwrap_or(returned).max(returned);
                let mut flags = 0;
                if generated > 1 {
                    flags |= FLAG_MULTIPLE_COMPLETIONS;
                }
                if request.stream {
                    flags |= FLAG_STREAMING;
                }
                Some(Self {
                    endpoint: "completion",
                    max_output_tokens: multiplied_limit(request.max_tokens, generated),
                    flags,
                })
            }
            RequestType::Messages(request) => {
                let mut flags = 0;
                if request
                    .tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty())
                {
                    flags |= FLAG_TOOLS;
                }
                if request.thinking.is_some() {
                    flags |= FLAG_REASONING;
                }
                if request.is_stream() {
                    flags |= FLAG_STREAMING;
                }
                Some(Self {
                    endpoint: "messages",
                    max_output_tokens: Some(request.max_tokens),
                    flags,
                })
            }
            RequestType::Responses(request) => {
                let mut flags = 0;
                if request
                    .tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty())
                {
                    flags |= FLAG_TOOLS;
                }
                if request.text.is_some() {
                    flags |= FLAG_STRUCTURED_OUTPUT;
                }
                if request.reasoning.is_some() {
                    flags |= FLAG_REASONING;
                }
                if request.stream.unwrap_or(false) {
                    flags |= FLAG_STREAMING;
                }
                Some(Self {
                    endpoint: "responses",
                    max_output_tokens: request.max_output_tokens,
                    flags,
                })
            }
            RequestType::Embedding(_) | RequestType::Classify(_) => None,
        }
    }
}

fn is_single_distribution_seed_sample(
    shape: GenerationShape,
    backend_request_count: usize,
    streaming_scope: StreamingDistributionSeedScope,
) -> bool {
    if backend_request_count != 1 || shape.flags & FLAG_MULTIPLE_COMPLETIONS != 0 {
        return false;
    }
    shape.flags & FLAG_STREAMING == 0
        || (shape.endpoint == "chat"
            && streaming_scope == StreamingDistributionSeedScope::DirectRegularChat)
}

fn multiplied_limit(per_completion: Option<u32>, multiplicity: u32) -> Option<u32> {
    per_completion.map(|limit| limit.saturating_mul(multiplicity))
}

fn trusted_header(ctx: &RequestContext, name: &str) -> Option<String> {
    ctx.input
        .headers
        .as_ref()
        .and_then(|headers| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn rejection_response(retry_after_secs: u32) -> Response {
    let mut response = error::create_error(
        StatusCode::TOO_MANY_REQUESTS,
        "adaptive_admission_saturated",
        "model fleet is temporarily saturated",
    );
    if let Ok(value) = HeaderValue::from_str(&retry_after_secs.max(1).to_string()) {
        response.headers_mut().insert(RETRY_AFTER, value);
    }
    response.extensions_mut().insert(LocalAdaptiveRejection);
    response
}

#[async_trait]
impl PipelineStage for AdaptiveAdmissionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let Some(controller) = ctx.components.adaptive_admission.clone() else {
            return Ok(None);
        };
        let Some(shape) = GenerationShape::for_request(&ctx.input.request_type) else {
            return Ok(None);
        };
        let prompt_tokens = ctx
            .state
            .preparation
            .as_ref()
            .map_or(0, |preparation| preparation.total_token_count());
        let partition = trusted_header(ctx, ADMISSION_PARTITION_HEADER)
            .unwrap_or_else(|| ctx.input.model_id.clone());
        let user = trusted_header(ctx, COMET_USER_HEADER)
            .or_else(|| {
                ctx.input
                    .tenant_request_meta
                    .as_ref()
                    .map(|meta| meta.tenant_key().as_str().to_string())
            })
            .unwrap_or_else(|| "anonymous".to_string());
        let workload_type = trusted_header(ctx, COMET_WORKLOAD_TYPE_HEADER)
            .unwrap_or_else(|| "unclassified".to_string());
        let tracker = controller.begin(
            partition,
            PredictionFeatures {
                model: ctx.input.model_id.clone(),
                user,
                workload_type,
                endpoint: shape.endpoint,
                prompt_tokens,
                max_output_tokens: shape.max_output_tokens,
                generation_flags: shape.flags,
            },
        );
        if tracker.should_reject() {
            let can_try_distribution_seed =
                matches!(
                    tracker.rejection_reason(),
                    Some("engine_waiting" | "running_limit")
                ) && ctx.state.preparation.as_ref().is_some_and(|preparation| {
                    is_single_distribution_seed_sample(
                        shape,
                        preparation.backend_request_count(),
                        ctx.input.streaming_distribution_seed_scope,
                    )
                }) && ctx
                    .input
                    .tenant_request_meta
                    .as_ref()
                    .and_then(|meta| meta.extension::<SchedulerAdmissionProof>())
                    .is_some();
            if can_try_distribution_seed {
                let Some(partition) = tracker.partition().map(str::to_string) else {
                    return Err(rejection_response(tracker.retry_after_secs()));
                };
                ctx.state.pending_distribution_seed = Some(PendingDistributionSeed {
                    partition,
                    retry_after_secs: tracker.retry_after_secs(),
                });
                ctx.state.adaptive_request = Some(tracker);
                return Ok(None);
            }
            return Err(rejection_response(tracker.retry_after_secs()));
        }
        ctx.state.adaptive_request = Some(tracker);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "AdaptiveAdmission"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::{
        chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
        messages::CreateMessageRequest, responses::ResponsesRequest,
    };

    use super::*;

    fn chat_shape(n: u32) -> GenerationShape {
        GenerationShape::for_request(&RequestType::Chat(Arc::new(ChatCompletionRequest {
            n: Some(n),
            ..Default::default()
        })))
        .unwrap()
    }

    fn generate_shape(n: u32) -> GenerationShape {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "text": "hello",
            "sampling_params": { "n": n }
        }))
        .unwrap();
        GenerationShape::for_request(&RequestType::Generate(Arc::new(request))).unwrap()
    }

    fn completion_shape(n: Option<u32>, best_of: Option<u32>) -> GenerationShape {
        let mut value = serde_json::json!({
            "model": "unknown",
            "prompt": "hello"
        });
        if let Some(n) = n {
            value["n"] = serde_json::json!(n);
        }
        if let Some(best_of) = best_of {
            value["best_of"] = serde_json::json!(best_of);
        }
        let request: CompletionRequest = serde_json::from_value(value).unwrap();
        GenerationShape::for_request(&RequestType::Completion(Arc::new(request))).unwrap()
    }

    #[test]
    fn scalar_generation_and_scalar_streaming_chat_can_enter_distribution_seed_path() {
        let mut streaming_chat = chat_shape(1);
        streaming_chat.flags |= FLAG_STREAMING;
        let cases = [
            ("scalar chat", chat_shape(1), 1, true),
            ("streaming chat", streaming_chat, 1, true),
            ("chat n", chat_shape(2), 1, false),
            ("generate n", generate_shape(2), 1, false),
            ("completion n", completion_shape(Some(2), None), 1, false),
            (
                "completion best_of",
                completion_shape(Some(1), Some(2)),
                1,
                false,
            ),
            (
                "multi-prompt completion fanout",
                completion_shape(Some(1), None),
                2,
                false,
            ),
        ];

        for (name, shape, backend_request_count, expected) in cases {
            assert_eq!(
                is_single_distribution_seed_sample(
                    shape,
                    backend_request_count,
                    StreamingDistributionSeedScope::DirectRegularChat,
                ),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn only_openai_chat_streaming_can_enter_distribution_seeding() {
        let requests = [
            RequestType::Chat(Arc::new(ChatCompletionRequest {
                stream: true,
                ..Default::default()
            })),
            RequestType::Generate(Arc::new(
                serde_json::from_value::<GenerateRequest>(serde_json::json!({
                    "model": "unknown",
                    "text": "hello",
                    "stream": true,
                }))
                .unwrap(),
            )),
            RequestType::Completion(Arc::new(
                serde_json::from_value::<CompletionRequest>(serde_json::json!({
                    "model": "unknown",
                    "prompt": "hello",
                    "stream": true,
                }))
                .unwrap(),
            )),
            RequestType::Messages(Arc::new(
                serde_json::from_value::<CreateMessageRequest>(serde_json::json!({
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 16,
                    "stream": true,
                }))
                .unwrap(),
            )),
            RequestType::Responses(Arc::new(
                serde_json::from_value::<ResponsesRequest>(serde_json::json!({
                    "model": "unknown",
                    "input": "hello",
                    "stream": true,
                }))
                .unwrap(),
            )),
        ];

        for (index, request) in requests.into_iter().enumerate() {
            let shape = GenerationShape::for_request(&request).unwrap();
            assert_ne!(shape.flags & FLAG_STREAMING, 0, "{}", shape.endpoint);
            assert_eq!(
                is_single_distribution_seed_sample(
                    shape,
                    1,
                    StreamingDistributionSeedScope::DirectRegularChat,
                ),
                index == 0,
                "unexpected streaming seed eligibility for {}",
                shape.endpoint
            );
        }
    }

    #[test]
    fn streaming_chat_fanout_remains_excluded() {
        let mut shape = chat_shape(2);
        shape.flags |= FLAG_STREAMING;

        assert!(!is_single_distribution_seed_sample(
            shape,
            1,
            StreamingDistributionSeedScope::DirectRegularChat,
        ));
        let mut scalar = chat_shape(1);
        scalar.flags |= FLAG_STREAMING;
        assert!(!is_single_distribution_seed_sample(
            scalar,
            2,
            StreamingDistributionSeedScope::DirectRegularChat,
        ));
    }

    #[test]
    fn streaming_chat_requires_direct_regular_chat_scope() {
        let mut streaming_chat = chat_shape(1);
        streaming_chat.flags |= FLAG_STREAMING;

        assert!(is_single_distribution_seed_sample(
            streaming_chat,
            1,
            StreamingDistributionSeedScope::DirectRegularChat,
        ));
        assert!(!is_single_distribution_seed_sample(
            streaming_chat,
            1,
            StreamingDistributionSeedScope::Ineligible,
        ));

        let non_streaming_chat = chat_shape(1);
        assert!(is_single_distribution_seed_sample(
            non_streaming_chat,
            1,
            StreamingDistributionSeedScope::Ineligible,
        ));
    }

    #[test]
    fn local_rejection_carries_internal_scheduler_marker() {
        let response = rejection_response(3);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "3");
        assert!(response
            .extensions()
            .get::<LocalAdaptiveRejection>()
            .is_some());
    }
}
