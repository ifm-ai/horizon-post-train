//! Terminal-success commit for cache-aware distribution seeds.

use async_trait::async_trait;
use axum::response::Response;

use super::{adaptive_admission::rejection_response, PipelineStage};
use crate::routers::{
    error,
    grpc::context::{FinalResponse, RequestContext, RequestType},
};

pub(crate) struct DistributionSeedCommitStage;

fn has_matching_terminal_response(
    request_type: &RequestType,
    final_response: Option<&FinalResponse>,
    has_responses_iteration_result: bool,
) -> bool {
    match request_type {
        RequestType::Chat(_) => matches!(final_response, Some(FinalResponse::Chat(_))),
        RequestType::Generate(_) => matches!(final_response, Some(FinalResponse::Generate(_))),
        RequestType::Completion(_) => {
            matches!(final_response, Some(FinalResponse::Completion(_)))
        }
        RequestType::Messages(_) => matches!(final_response, Some(FinalResponse::Messages(_))),
        RequestType::Responses(_) => has_responses_iteration_result,
        RequestType::Embedding(_) | RequestType::Classify(_) => false,
    }
}

#[async_trait]
impl PipelineStage for DistributionSeedCommitStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let has_seed = ctx
            .state
            .load_guards
            .as_ref()
            .is_some_and(|guards| guards.has_distribution_seed());
        if !has_seed {
            return Ok(None);
        }
        if ctx.is_streaming() {
            return Err(rejection_response(1));
        }
        if !has_matching_terminal_response(
            &ctx.input.request_type,
            ctx.state.response.final_response.as_ref(),
            ctx.state.response.responses_iteration_result.is_some(),
        ) {
            return Err(error::internal_error(
                "distribution_seed_terminal_response_missing",
                "Distribution seed did not produce the expected terminal response",
            ));
        }
        if let Some(guards) = ctx.state.load_guards.as_mut() {
            guards.commit_distribution_seed_success();
        }
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "DistributionSeedCommit"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use axum::http::StatusCode;
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        completion::{CompletionRequest, CompletionResponse},
        generate::GenerateRequest,
        messages::{CreateMessageRequest, Message},
        responses::ResponsesRequest,
    };
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::json;
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{
        routers::grpc::context::{LoadGuards, SharedComponents},
        worker::WorkerRegistry,
    };

    fn components() -> Arc<SharedComponents> {
        Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            multimodal: None,
            adaptive_admission: None,
        })
    }

    fn generate_context(stream: bool) -> RequestContext {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "kimi-k3",
            "text": "hello",
            "stream": stream,
        }))
        .expect("generate request");
        RequestContext::for_generate(Arc::new(request), None, "kimi-k3".to_string(), components())
    }

    fn test_seed(ctx: &mut RequestContext) -> Arc<AtomicBool> {
        let committed = Arc::new(AtomicBool::new(false));
        ctx.state.load_guards = Some(LoadGuards::test_distribution_seed(Arc::clone(&committed)));
        committed
    }

    #[test]
    fn terminal_response_match_is_endpoint_exact() {
        let chat = RequestType::Chat(Arc::new(
            serde_json::from_value::<ChatCompletionRequest>(json!({
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "hello"}],
            }))
            .expect("chat request"),
        ));
        let generate = RequestType::Generate(Arc::new(
            serde_json::from_value::<GenerateRequest>(json!({
                "model": "kimi-k3",
                "text": "hello",
            }))
            .expect("generate request"),
        ));
        let completion = RequestType::Completion(Arc::new(
            serde_json::from_value::<CompletionRequest>(json!({
                "model": "kimi-k3",
                "prompt": "hello",
            }))
            .expect("completion request"),
        ));
        let messages = RequestType::Messages(Arc::new(
            serde_json::from_value::<CreateMessageRequest>(json!({
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            }))
            .expect("messages request"),
        ));
        let responses = RequestType::Responses(Arc::new(
            serde_json::from_value::<ResponsesRequest>(json!({
                "model": "kimi-k3",
                "input": "hello",
            }))
            .expect("responses request"),
        ));

        let chat_final = FinalResponse::Chat(
            serde_json::from_value::<ChatCompletionResponse>(json!({
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": "kimi-k3",
                "choices": [],
            }))
            .expect("chat response"),
        );
        let generate_final = FinalResponse::Generate(Vec::new());
        let completion_final = FinalResponse::Completion(
            serde_json::from_value::<CompletionResponse>(json!({
                "id": "completion-1",
                "object": "text_completion",
                "created": 1,
                "model": "kimi-k3",
                "choices": [],
            }))
            .expect("completion response"),
        );
        let messages_final = FinalResponse::Messages(
            serde_json::from_value::<Message>(json!({
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "kimi-k3",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }))
            .expect("messages response"),
        );

        assert!(has_matching_terminal_response(
            &chat,
            Some(&chat_final),
            false
        ));
        assert!(has_matching_terminal_response(
            &generate,
            Some(&generate_final),
            false
        ));
        assert!(has_matching_terminal_response(
            &completion,
            Some(&completion_final),
            false
        ));
        assert!(has_matching_terminal_response(
            &messages,
            Some(&messages_final),
            false
        ));
        assert!(has_matching_terminal_response(&responses, None, true));

        assert!(!has_matching_terminal_response(
            &chat,
            Some(&generate_final),
            false
        ));
        assert!(!has_matching_terminal_response(&generate, None, false));
        assert!(!has_matching_terminal_response(&responses, None, false));
    }

    #[tokio::test]
    async fn no_seed_is_a_noop_without_terminal_state() {
        let mut ctx = generate_context(false);
        let result = DistributionSeedCommitStage.execute(&mut ctx).await;
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn matching_non_streaming_terminal_response_commits_seed() {
        let mut ctx = generate_context(false);
        let committed = test_seed(&mut ctx);
        ctx.state.response.final_response = Some(FinalResponse::Generate(Vec::new()));

        let result = DistributionSeedCommitStage.execute(&mut ctx).await;

        assert!(matches!(result, Ok(None)));
        assert!(committed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn missing_or_mismatched_terminal_state_does_not_commit() {
        let mut missing = generate_context(false);
        let missing_committed = test_seed(&mut missing);
        let response = DistributionSeedCommitStage
            .execute(&mut missing)
            .await
            .expect_err("missing terminal response must fail closed");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!missing_committed.load(Ordering::Acquire));

        let mut mismatched = generate_context(false);
        let mismatched_committed = test_seed(&mut mismatched);
        mismatched.state.response.final_response = Some(FinalResponse::Completion(
            serde_json::from_value::<CompletionResponse>(json!({
                "id": "completion-1",
                "object": "text_completion",
                "created": 1,
                "model": "kimi-k3",
                "choices": [],
            }))
            .expect("completion response"),
        ));
        let response = DistributionSeedCommitStage
            .execute(&mut mismatched)
            .await
            .expect_err("mismatched terminal response must fail closed");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!mismatched_committed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn streaming_seed_is_rejected_without_commit() {
        let mut ctx = generate_context(true);
        let committed = test_seed(&mut ctx);
        ctx.state.response.final_response = Some(FinalResponse::Generate(Vec::new()));

        let response = DistributionSeedCommitStage
            .execute(&mut ctx)
            .await
            .expect_err("streaming seeds are unsupported in the first release");

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(!committed.load(Ordering::Acquire));
    }
}
