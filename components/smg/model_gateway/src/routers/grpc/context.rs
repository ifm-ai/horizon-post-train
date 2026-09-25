//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::http::HeaderMap;
use llm_tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::{ClassifyRequest, ClassifyResponse},
    completion::{CompletionRequest, CompletionResponse},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    generate::{GenerateRequest, GenerateResponse},
    messages::{CreateMessageRequest, Message},
    responses::ResponsesRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::debug;

use super::{
    adaptive_admission::{
        AdaptiveAdmissionController, AdaptiveRequestTracker, DistributionHeadroomLease,
        OrdinaryDistributionDispatchGuard,
    },
    client::GrpcClient,
    common::stages::encode::EncodeDispatchPlan,
    multimodal::{MultimodalComponents, MultimodalIntermediate},
    proto_wrapper::{
        EncodeItemBootstrapInfo, ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest,
        ProtoRequest, ProtoStream,
    },
};
use crate::{
    middleware::{scheduler::ClaimedSchedulerAdmissionProof, TenantRequestMeta},
    policies::{LoadBalancingPolicy, OwnerPressureDispatchPlan},
    worker::{RuntimeType, Worker, WorkerLoadGuard, WorkerRegistry},
};

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub(crate) struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    /// Canonical model ID used after aliases are resolved at request entry.
    pub model_id: String,
    pub tenant_request_meta: Option<TenantRequestMeta>,
    /// Trusted pipeline-derived scope for the streaming distribution-seed path.
    /// Protocol shape alone cannot distinguish direct Chat Completions from
    /// internal Responses iterations or Harmony chat requests.
    pub streaming_distribution_seed_scope: StreamingDistributionSeedScope,
}

/// Only a direct scalar streaming request through the Regular Chat Completions
/// pipeline may use the streaming distribution-seed lifecycle. All other
/// request construction paths default to `Ineligible`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum StreamingDistributionSeedScope {
    #[default]
    Ineligible,
    DirectRegularChat,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub(crate) enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Completion(Arc<CompletionRequest>),
    Responses(Arc<ResponsesRequest>),
    Embedding(Arc<EmbeddingRequest>),
    Classify(Arc<ClassifyRequest>),
    Messages(Arc<CreateMessageRequest>),
}

impl RequestType {
    /// Overwrite the request's own `model` field.
    ///
    /// Callers hold the request behind an `Arc` that the retry loop also
    /// holds, so `Arc::make_mut` copies the request here. That cost is paid
    /// only on the alias path — [`RequestContext::new`] skips this call
    /// entirely when the client already used the canonical model ID.
    fn set_model(&mut self, model_id: &str) {
        fn replace(model: &mut String, model_id: &str) {
            model.clear();
            model.push_str(model_id);
        }

        match self {
            Self::Chat(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Generate(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Completion(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Responses(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Embedding(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Classify(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Messages(request) => replace(&mut Arc::make_mut(request).model, model_id),
        }
    }

    /// Client-supplied backend request id (`rid`), where the protocol carries
    /// one. Responses ids are storage-owned (`resp_*`) and never client-set.
    pub fn rid(&self) -> Option<&str> {
        match self {
            Self::Chat(r) => r.rid.as_deref(),
            Self::Generate(r) => r.rid.as_deref(),
            Self::Completion(r) => r.rid.as_deref(),
            Self::Embedding(r) => r.rid.as_deref(),
            Self::Classify(r) => r.rid.as_deref(),
            Self::Messages(r) => r.rid.as_deref(),
            Self::Responses(_) => None,
        }
    }
}

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Responses(_) => write!(f, "Responses"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

impl std::fmt::Display for FinalResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

/// Shared components (injected once at creation)
pub(crate) struct SharedComponents {
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    /// Configured tool parser name (from CLI `--tool-call-parser`)
    pub configured_tool_parser: Option<String>,
    /// Configured reasoning parser name (from CLI `--reasoning-parser`)
    pub configured_reasoning_parser: Option<String>,
    /// Multimodal processing components (initialized at router creation)
    pub multimodal: Option<Arc<MultimodalComponents>>,
    pub adaptive_admission: Option<Arc<AdaptiveAdmissionController>>,
}

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// Owned here rather than inside `PreparationOutput` so EPD's `EncodeStage`
    /// can borrow it for the with-pixels encode serialization before request
    /// building `take()`s it for the prefill serialization.
    pub multimodal_intermediate: Option<MultimodalIntermediate>,

    /// `Some` iff the request is multimodal EPD and worker selection produced
    /// encode assignments. Request building injects the bootstrap info and drops
    /// prefill pixels; request execution `take()`s the dispatch plan.
    pub encode_outputs: Option<EncodeOutputs>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    /// Predicted token-work reservation. Non-streaming response processing
    /// completes it from the final usage counters; streaming processing moves
    /// it into the background stream task. Any earlier error drops it and
    /// releases predicted outstanding work without training on a partial run.
    pub adaptive_request: Option<AdaptiveRequestTracker>,

    /// An aggregate adaptive rejection that may proceed only through the
    /// cache-aware, scheduler-authorized clean-peer recovery path in worker
    /// selection. No ordinary policy selection is allowed while this is set.
    pub pending_distribution_seed: Option<PendingDistributionSeed>,

    /// Non-cloneable authorization and target-capacity guards for an exact
    /// clean-peer dispatch. Moved into the request-lifetime load guard before
    /// backend execution; every earlier failure releases both by `Drop`.
    pub distribution_seed_guard: Option<DistributionSeedDispatchGuard>,

    /// Exact ordinary target claim held until request execution installs the
    /// worker load guard. This closes the worker-selection-to-dispatch window
    /// in which a clean-peer seed might otherwise reserve the same target.
    pub ordinary_distribution_guard: Option<OrdinaryDistributionDispatchGuard>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    /// Router-local work reserved atomically by the regular gRPC policy.
    /// Kept separately from `WorkerSelection` so an early pipeline failure
    /// still releases it when the request context is dropped.
    pub policy_reservation: Option<PolicyReservation>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Stage 4: Request building outputs
    pub execution_plan: Option<ExecutionPlan>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Load guard for worker load tracking (created at execution stage)
    pub load_guards: Option<LoadGuards>,

    // Stage 6: Response processing state
    pub response: ResponseState,
}

#[derive(Debug)]
pub(crate) struct PendingDistributionSeed {
    pub(crate) partition: String,
    pub(crate) retry_after_secs: u32,
}

#[derive(Debug)]
pub(crate) struct DistributionSeedDispatchGuard {
    pub(crate) policy_plan: OwnerPressureDispatchPlan,
    pub(crate) _scheduler_proof: ClaimedSchedulerAdmissionProof,
    // Keep this field last. Rust drops fields in declaration order, so the
    // prefix plan must enter cooldown before model-wide target suppression is
    // released by the headroom lease.
    pub(crate) headroom: DistributionHeadroomLease,
    pub(crate) retry_after_secs: u32,
}

impl DistributionSeedDispatchGuard {
    fn commit_success(&mut self) {
        self.policy_plan.commit_success();
    }
}

/// Per-item bootstrap rendezvous info for prefill, plus the dispatch plan that
/// fans out to encode workers.
///
/// Not `#[derive(Debug)]`: `EncodeDispatchPlan` transitively holds
/// non-`Debug` raw proto payloads (`TokenSpeedMultimodalItem`).
///
/// Owns the encode jobs' SHM/RDMA Drop guards: dropping this before request
/// execution dispatches (early return / cancellation) reclaims the staged
/// `/dev/shm` segments via `PreparedEncodeItem`'s `Drop`.
pub(crate) struct EncodeOutputs {
    pub bootstrap_info: Vec<EncodeItemBootstrapInfo>,
    pub dispatch: EncodeDispatchPlan,
}

/// Execution shape produced by request building and consumed by request execution.
pub(crate) enum ExecutionPlan {
    Single(ProtoRequest),
    PrefillDecode(ProtoGenerateRequest),
    EncodePrefillDecode {
        request: ProtoGenerateRequest,
    },
    /// Batched completion fan-out: one backend request per prompt, all
    /// dispatched with the disaggregation shape given by `kind`. Sub-request
    /// ids are `{shared_request_id}-p{i}`; the client-visible response id is
    /// `shared_request_id`.
    Batch {
        kind: ExecutionPlanKind,
        shared_request_id: String,
        requests: Vec<ProtoGenerateRequest>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPlanKind {
    Single,
    PrefillDecode,
    EncodePrefillDecode,
}

impl ExecutionPlan {
    pub(crate) fn generate(kind: ExecutionPlanKind, request: ProtoGenerateRequest) -> Self {
        match kind {
            ExecutionPlanKind::Single => Self::Single(ProtoRequest::Generate(request)),
            ExecutionPlanKind::PrefillDecode => Self::PrefillDecode(request),
            ExecutionPlanKind::EncodePrefillDecode => Self::EncodePrefillDecode { request },
        }
    }

    pub(crate) fn embed(request: ProtoEmbedRequest) -> Self {
        Self::Single(ProtoRequest::Embed(request))
    }

    pub(crate) fn request_id(&self) -> &str {
        match self {
            Self::Single(request) => request.request_id(),
            Self::PrefillDecode(request) | Self::EncodePrefillDecode { request, .. } => {
                request.request_id()
            }
            Self::Batch {
                shared_request_id, ..
            } => shared_request_id,
        }
    }

    pub(crate) fn request_type(&self) -> &'static str {
        match self {
            Self::Single(ProtoRequest::Generate(_))
            | Self::PrefillDecode(_)
            | Self::EncodePrefillDecode { .. }
            | Self::Batch { .. } => "generate",
            Self::Single(ProtoRequest::Embed(_)) => "embed",
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        match self {
            Self::Single(_) => "single",
            Self::PrefillDecode(_) => "prefill_decode",
            Self::EncodePrefillDecode { .. } => "encode_prefill_decode",
            Self::Batch { kind, .. } => match kind {
                ExecutionPlanKind::Single => "single",
                ExecutionPlanKind::PrefillDecode => "prefill_decode",
                ExecutionPlanKind::EncodePrefillDecode => "encode_prefill_decode",
            },
        }
    }
}

/// Output from preparation stage (Step 1)
///
/// Each request type produces its own variant, eliminating optional fields
/// that are always None for certain pipelines.
pub(crate) enum PreparationOutput {
    Chat {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    Messages {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    Completion {
        /// One entry per prompt; scalar requests carry exactly one.
        items: Vec<CompletionItem>,
        /// `Some` iff multiple prompts: their texts joined for routing,
        /// mirroring the HTTP router's `extract_text_for_routing`.
        joined_routing_text: Option<String>,
    },
    Generate {
        original_text: Option<String>,
        token_ids: Vec<u32>,
    },
    Embedding {
        original_text: String,
        token_ids: Vec<u32>,
    },
    Harmony {
        token_ids: Vec<u32>,
        selection_text: String,
        tool_constraints: Option<(String, String)>,
        /// Request with response_format cleared (when converted to structural tag)
        modified_request: Option<Box<ChatCompletionRequest>>,
        #[expect(dead_code, reason = "stored for future Harmony history tracking")]
        harmony_messages: Vec<super::harmony::HarmonyMessage>,
        harmony_stop_ids: Vec<u32>,
    },
}

/// One tokenized completion prompt.
pub(crate) struct CompletionItem {
    pub text: String,
    pub token_ids: Vec<u32>,
}

impl PreparationOutput {
    /// Number of independent backend requests represented by this prepared
    /// input. A single scheduler proof and headroom lease may never authorize
    /// a batched completion fan-out.
    pub(crate) fn backend_request_count(&self) -> usize {
        match self {
            Self::Completion { items, .. } => items.len(),
            _ => 1,
        }
    }

    /// Total prompt tokens represented by this request. Completion batches
    /// sum every prompt because they fan out into independent backend work.
    pub fn total_token_count(&self) -> u32 {
        let count = match self {
            Self::Completion { items, .. } => {
                items.iter().map(|item| item.token_ids.len()).sum::<usize>()
            }
            _ => self.token_ids().len(),
        };
        u32::try_from(count).unwrap_or(u32::MAX)
    }

    /// Token IDs (common to all variants). Batched completions expose the
    /// first prompt's tokens as the routing-affinity proxy.
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Chat { token_ids, .. }
            | Self::Messages { token_ids, .. }
            | Self::Generate { token_ids, .. }
            | Self::Embedding { token_ids, .. }
            | Self::Harmony { token_ids, .. } => token_ids,
            Self::Completion { items, .. } => {
                items.first().map_or(&[], |item| item.token_ids.as_slice())
            }
        }
    }

    /// Text for worker routing: original_text for regular pipelines, selection_text for Harmony.
    /// Chat/Messages borrow from processed_messages.text to avoid a redundant clone.
    pub fn routing_text(&self) -> Option<&str> {
        match self {
            Self::Chat {
                processed_messages, ..
            }
            | Self::Messages {
                processed_messages, ..
            } => Some(&processed_messages.text),
            Self::Completion {
                items,
                joined_routing_text,
            } => joined_routing_text
                .as_deref()
                .or_else(|| items.first().map(|item| item.text.as_str())),
            Self::Embedding { original_text, .. } => Some(original_text),
            Self::Generate { original_text, .. } => original_text.as_deref(),
            Self::Harmony { selection_text, .. } => Some(selection_text),
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncodeWorkerAssignment {
    pub item_index: usize,
    pub worker: Arc<dyn Worker>,
}

/// Worker selection (Step 2)
pub(crate) enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    /// Disaggregated prefill/decode selection. EPD layers per-item encode
    /// assignments on top; plain PD leaves `encode_assignments` unset.
    Disaggregated {
        encode_assignments: Option<Vec<EncodeWorkerAssignment>>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        runtime_type: RuntimeType,
    },
}

/// A policy reservation tied to the request lifetime.
///
/// Selection increments the policy's router-local work synchronously. Moving
/// this guard into `LoadGuards` keeps it until the backend request finishes;
/// leaving it in `ProcessingState` releases it on any earlier error path.
pub(crate) struct PolicyReservation {
    policy: Arc<dyn LoadBalancingPolicy>,
    worker_url: String,
    cost: u64,
}

impl PolicyReservation {
    pub fn new(
        policy: Arc<dyn LoadBalancingPolicy>,
        worker_url: impl Into<String>,
        cost: u64,
    ) -> Self {
        Self {
            policy,
            worker_url: worker_url.into(),
            cost,
        }
    }
}

impl Drop for PolicyReservation {
    fn drop(&mut self) {
        self.policy.release_reservation(&self.worker_url, self.cost);
    }
}

/// Client selection (Step 3)
#[derive(Clone)]
pub(crate) enum ClientSelection {
    Single {
        client: GrpcClient,
    },
    /// Disaggregated prefill/decode scheduler clients. EPD encode workers are
    /// contacted directly from `WorkerSelection::Disaggregated` assignments.
    Disaggregated {
        prefill: GrpcClient,
        decode: GrpcClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub(crate) struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
}

/// Load guards for worker load tracking
/// Automatically decrements load when dropped
#[cfg(test)]
pub(crate) struct TestLoadGuardsDropProbe(Arc<AtomicBool>);

#[cfg(test)]
impl Drop for TestLoadGuardsDropProbe {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(crate) enum LoadGuards {
    Single {
        _guard: WorkerLoadGuard,
        _policy_reservation: Option<PolicyReservation>,
        distribution_seed: Option<DistributionSeedDispatchGuard>,
    },
    /// Disaggregated guards cover the prefill+decode pair. EPD encode workers are
    /// assigned per item; their fire-and-supervise RPCs do not hold load guards.
    Disaggregated {
        _prefill: WorkerLoadGuard,
        _decode: WorkerLoadGuard,
    },
    /// Batched completion fan-out: one guard set per sub-request so load-aware
    /// policies see the real backend concurrency.
    Batch {
        _guards: Vec<LoadGuards>,
        _policy_reservation: Option<PolicyReservation>,
        distribution_seed: Option<DistributionSeedDispatchGuard>,
    },
    /// Test-only terminal-success probe for the distribution-seed commit stage.
    /// Production construction still requires the real scheduler proof, policy
    /// reservation, and adaptive headroom lease.
    #[cfg(test)]
    TestDistributionSeed {
        committed: Arc<AtomicBool>,
        _drop_probe: Option<TestLoadGuardsDropProbe>,
    },
}

impl LoadGuards {
    pub(crate) fn has_distribution_seed(&self) -> bool {
        match self {
            Self::Single {
                distribution_seed, ..
            }
            | Self::Batch {
                distribution_seed, ..
            } => distribution_seed.is_some(),
            Self::Disaggregated { .. } => false,
            #[cfg(test)]
            Self::TestDistributionSeed { .. } => true,
        }
    }

    pub(crate) fn commit_distribution_seed_success(&mut self) {
        match self {
            Self::Single {
                distribution_seed, ..
            }
            | Self::Batch {
                distribution_seed, ..
            } => {
                if let Some(seed) = distribution_seed {
                    seed.commit_success();
                }
            }
            Self::Disaggregated { .. } => {}
            #[cfg(test)]
            Self::TestDistributionSeed { committed, .. } => {
                committed.store(true, Ordering::Release);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn test_distribution_seed(committed: Arc<AtomicBool>) -> Self {
        Self::TestDistributionSeed {
            committed,
            _drop_probe: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_distribution_seed_lifecycle(
        committed: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    ) -> Self {
        Self::TestDistributionSeed {
            committed,
            _drop_probe: Some(TestLoadGuardsDropProbe(dropped)),
        }
    }

    pub fn new(
        selection: &WorkerSelection,
        headers: Option<&HeaderMap>,
        policy_reservation: Option<PolicyReservation>,
        distribution_seed: Option<DistributionSeedDispatchGuard>,
    ) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: WorkerLoadGuard::new(worker.clone(), headers),
                _policy_reservation: policy_reservation,
                distribution_seed,
            },
            WorkerSelection::Disaggregated {
                prefill, decode, ..
            } => {
                debug_assert!(policy_reservation.is_none());
                debug_assert!(distribution_seed.is_none());
                LoadGuards::Disaggregated {
                    _prefill: WorkerLoadGuard::new(prefill.clone(), headers),
                    _decode: WorkerLoadGuard::new(decode.clone(), headers),
                }
            }
        }
    }

    /// One guard set per concurrent sub-request.
    pub fn scaled(
        selection: &WorkerSelection,
        headers: Option<&HeaderMap>,
        count: usize,
        policy_reservation: Option<PolicyReservation>,
        distribution_seed: Option<DistributionSeedDispatchGuard>,
    ) -> Self {
        if count <= 1 {
            Self::new(selection, headers, policy_reservation, distribution_seed)
        } else {
            Self::Batch {
                _guards: (0..count)
                    .map(|_| Self::new(selection, headers, None, None))
                    .collect(),
                _policy_reservation: policy_reservation,
                distribution_seed,
            }
        }
    }
}

/// Response processing state (Step 6)
#[derive(Default)]
pub(crate) struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// Derived skip_special_tokens for streaming (set in preparation, read in response_processing).
    /// Stored here because PreparationOutput is consumed by request_building before
    /// response_processing runs.
    pub skip_special_tokens: Option<bool>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,
}

impl RequestContext {
    /// Build a context, resolving a model alias to its canonical model ID.
    ///
    /// This is the single place the gRPC pipeline canonicalizes. Both
    /// `input.model_id` and the request's own `model` field are rewritten, so
    /// every stage below — worker selection, tokenizer lookup, parser
    /// selection, tool call ID format — reads the canonical ID without
    /// resolving anything itself.
    ///
    /// One visible consequence: the response reports the canonical model ID,
    /// not the alias the client sent. That matches how the OpenAI API answers
    /// with the model it actually ran.
    fn new(
        mut request_type: RequestType,
        headers: Option<HeaderMap>,
        mut model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        if let Some(canonical_model_id) = components.worker_registry.resolve_model_alias(&model_id)
        {
            model_id.clear();
            model_id.push_str(&canonical_model_id);
            request_type.set_model(&model_id);
        }
        Self {
            input: RequestInput {
                request_type,
                headers,
                model_id,
                tenant_request_meta: None,
                streaming_distribution_seed_scope: StreamingDistributionSeedScope::Ineligible,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        streaming_distribution_seed_scope: StreamingDistributionSeedScope,
    ) -> Self {
        let mut ctx = Self::new(RequestType::Chat(request), headers, model_id, components);
        ctx.input.streaming_distribution_seed_scope = streaming_distribution_seed_scope;
        ctx
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Generate(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for completion request
    pub fn for_completion(
        request: Arc<CompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Completion(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Responses(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for embedding request
    pub fn for_embedding(
        request: Arc<EmbeddingRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Embedding(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for classify request
    pub fn for_classify(
        request: Arc<ClassifyRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Classify(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for messages request
    pub fn for_messages(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Messages(request),
            headers,
            model_id,
            components,
        )
    }

    /// Get chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request(&self) -> &ChatCompletionRequest {
        match &self.input.request_type {
            RequestType::Chat(req) => req.as_ref(),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clone of chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get completion request (panics if not completion)
    #[expect(
        dead_code,
        reason = "ref accessor provided for API completeness alongside Arc accessor"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request(&self) -> &CompletionRequest {
        match &self.input.request_type {
            RequestType::Completion(req) => req.as_ref(),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of completion request (panics if not completion)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request_arc(&self) -> Arc<CompletionRequest> {
        match &self.input.request_type {
            RequestType::Completion(req) => Arc::clone(req),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Get messages request (panics if not messages)
    #[expect(
        dead_code,
        reason = "scaffolding for Messages API pipeline, wired in follow-up PR"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request(&self) -> &CreateMessageRequest {
        match &self.input.request_type {
            RequestType::Messages(req) => req.as_ref(),
            _ => panic!("Expected messages request"),
        }
    }

    /// Get Arc clone of messages request (panics if not messages)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request_arc(&self) -> Arc<CreateMessageRequest> {
        match &self.input.request_type {
            RequestType::Messages(req) => Arc::clone(req),
            _ => panic!("Expected messages request"),
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Completion(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
            RequestType::Messages(req) => req.stream.unwrap_or(false),
            RequestType::Embedding(_) => false, // Embeddings are never streaming
            RequestType::Classify(_) => false,  // Classification is never streaming
        }
    }

    /// Get the cached tokenizer, cloning the Arc (cheap 8-byte clone)
    ///
    /// Returns None if tokenizer hasn't been resolved yet.
    /// The tokenizer is resolved once in the preparation stage and cached for reuse.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.state.tokenizer.clone()
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl WorkerSelection {
    pub fn is_disaggregated(&self) -> bool {
        matches!(self, Self::Disaggregated { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            Self::Disaggregated { .. } => None,
        }
    }

    /// Record circuit breaker outcome for all workers based on HTTP status code.
    pub fn record_outcome(&self, status_code: u16) {
        match self {
            Self::Single { worker } => worker.record_outcome(status_code),
            Self::Disaggregated {
                prefill, decode, ..
            } => {
                // EPD encode dispatch is asynchronous and supervised by
                // RequestExecution; this records only the prefill/decode leg.
                prefill.record_outcome(status_code);
                decode.record_outcome(status_code);
            }
        }
    }

    /// Record circuit breaker outcomes for disaggregated dispatch (individual tracking)
    pub fn record_prefill_decode_outcomes(&self, prefill_status: u16, decode_status: u16) {
        if let Self::Disaggregated {
            prefill, decode, ..
        } = self
        {
            prefill.record_outcome(prefill_status);
            decode.record_outcome(decode_status);
        }
    }

    /// Record circuit breaker outcome for prefill worker only (sequential PD)
    pub fn record_outcome_prefill(&self, status_code: u16) {
        match self {
            Self::Disaggregated { prefill, .. } => {
                prefill.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_prefill called on Single worker selection, ignoring");
            }
        }
    }

    /// Record circuit breaker outcome for decode worker only (sequential PD)
    pub fn record_outcome_decode(&self, status_code: u16) {
        match self {
            Self::Disaggregated { decode, .. } => {
                decode.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_decode called on Single worker selection, ignoring");
            }
        }
    }

    #[expect(clippy::type_complexity)]
    pub fn disaggregated_pair(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Disaggregated {
                prefill, decode, ..
            } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    /// Get the runtime type for disaggregated mode.
    pub fn disaggregated_runtime_type(&self) -> Option<&RuntimeType> {
        match self {
            Self::Disaggregated { runtime_type, .. } => Some(runtime_type),
            Self::Single { .. } => None,
        }
    }

    pub fn encode_assignments(&self) -> Option<&[EncodeWorkerAssignment]> {
        match self {
            Self::Disaggregated {
                encode_assignments, ..
            } => encode_assignments.as_deref(),
            Self::Single { .. } => None,
        }
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl ClientSelection {
    pub fn single(&self) -> Option<&GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn disaggregated_mut(&mut self) -> Option<(&mut GrpcClient, &mut GrpcClient)> {
        match self {
            Self::Disaggregated { prefill, decode } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
        /// PD timing context, for honest PD TTFT (prefill start to first decode token).
        pd_timing: PdTiming,
    },
    /// Embedding requests return a single response, not a stream
    Embedding {
        response: ProtoEmbedComplete,
    },
    /// Batched completion fan-out: one result per prompt, in prompt order.
    Batch {
        results: Vec<ExecutionResult>,
    },
}

/// Timing context threaded from PD execution into the streaming layer so the
/// first decode token can be measured against prefill start.
#[derive(Clone)]
pub(crate) struct PdTiming {
    /// Monotonic instant the prefill RPC was dispatched.
    pub prefill_start: std::time::Instant,
    /// Backend runtime label (e.g. "sglang", "vllm") for the PD metric set.
    pub runtime: &'static str,
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Completion response (OpenAI /v1/completions format)
    Completion(CompletionResponse),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
    /// Messages API response
    Messages(Message),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion_prep(texts: &[&str], joined: Option<&str>) -> PreparationOutput {
        PreparationOutput::Completion {
            items: texts
                .iter()
                .enumerate()
                .map(|(i, text)| CompletionItem {
                    text: (*text).to_string(),
                    token_ids: vec![i as u32],
                })
                .collect(),
            joined_routing_text: joined.map(str::to_string),
        }
    }

    #[test]
    fn completion_preparation_routes_by_first_item_or_joined_text() {
        let scalar = completion_prep(&["hello"], None);
        assert_eq!(scalar.routing_text(), Some("hello"));
        assert_eq!(scalar.token_ids(), &[0]);

        let batch = completion_prep(&["a", "b"], Some("a b"));
        assert_eq!(batch.routing_text(), Some("a b"));
        assert_eq!(batch.token_ids(), &[0]);
    }

    #[test]
    fn batch_execution_plan_reports_shared_id_and_kind_label() {
        let plan = ExecutionPlan::Batch {
            kind: ExecutionPlanKind::PrefillDecode,
            shared_request_id: "cmpl_shared".to_string(),
            requests: vec![],
        };
        assert_eq!(plan.request_id(), "cmpl_shared");
        assert_eq!(plan.request_type(), "generate");
        assert_eq!(plan.mode_label(), "prefill_decode");
    }
}
