//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use tracing::{error, warn};

use super::{adaptive_admission::rejection_response, PipelineStage};
use crate::{
    middleware::scheduler::SchedulerAdmissionProof,
    observability::metrics::{metrics_labels, Metrics},
    policies::{
        CacheAwarePolicy, CacheAwareSelection, DistributionProtectionSnapshot, LoadBalancingPolicy,
        PolicyRegistry, SeedWorkerHeadroom, SelectWorkerInfo, WorkerLeg,
    },
    routers::{
        common::header_utils::worker_url_is_allowed,
        error,
        grpc::{
            adaptive_admission::{AdaptiveAdmissionController, OrdinaryDistributionDispatchGuard},
            context::{
                DistributionSeedDispatchGuard, EncodeWorkerAssignment, PendingDistributionSeed,
                PolicyReservation, RequestContext, RequestType, StreamingDistributionSeedScope,
                WorkerSelection,
            },
            multimodal,
        },
    },
    worker::{
        ConnectionMode, HashRing, RuntimeType, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
};

/// Result type for PD worker pair selection: (prefill, decode, runtime_type)
type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);
type SingleWorkerSelection = (
    Arc<dyn Worker>,
    Option<PolicyReservation>,
    Option<OrdinaryDistributionDispatchGuard>,
);

/// Result type for EPD worker selection: (encode assignments, prefill, decode, runtime_type).
type EncodePrefillDecodeWorkerSelection = (
    Vec<EncodeWorkerAssignment>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    RuntimeType,
);

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
    /// EPD mode: select encode + prefill + decode workers
    EncodePrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
        }
    }
}

fn is_cache_credit_chat_eligible(
    mode: WorkerSelectionMode,
    seed_scope: StreamingDistributionSeedScope,
    backend_request_count: usize,
    request_type: &RequestType,
) -> bool {
    mode == WorkerSelectionMode::Regular
        && seed_scope == StreamingDistributionSeedScope::DirectRegularChat
        && backend_request_count == 1
        && matches!(
            request_type,
            RequestType::Chat(request) if request.stream && request.n.unwrap_or(1) == 1
        )
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let pending_distribution_seed = ctx.state.pending_distribution_seed.take();
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let intermediate = ctx.state.multimodal_intermediate.as_ref();

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();

        let model_id = ctx.input.model_id.as_str();
        let cache_credit_chat_eligible = is_cache_credit_chat_eligible(
            self.mode,
            ctx.input.streaming_distribution_seed_scope,
            prep.backend_request_count(),
            &ctx.input.request_type,
        );
        let distribution_scope = ctx.state.adaptive_request.as_ref().and_then(|tracker| {
            let partition = tracker.partition()?;
            let controller = ctx.components.adaptive_admission.as_ref()?;
            controller
                .distribution_headroom_enabled(partition)
                .then(|| (Arc::clone(controller), partition.to_string()))
        });
        let workers = if let Some(pending) = pending_distribution_seed {
            if self.mode != WorkerSelectionMode::Regular {
                return Err(rejection_response(pending.retry_after_secs));
            }
            let Some((worker, guard)) =
                self.select_distribution_seed(ctx, &pending, model_id, text, tokens, headers)
            else {
                return Err(rejection_response(pending.retry_after_secs));
            };
            ctx.state.distribution_seed_guard = Some(guard);
            WorkerSelection::Single { worker }
        } else {
            match self.mode {
                WorkerSelectionMode::Regular => {
                    match self.select_single_worker(
                        model_id,
                        text,
                        tokens,
                        headers,
                        distribution_scope
                            .as_ref()
                            .map(|(controller, partition)| (controller, partition.as_str())),
                        cache_credit_chat_eligible,
                    )? {
                        Some((worker, reservation, ordinary_distribution_guard)) => {
                            ctx.state.policy_reservation = reservation;
                            ctx.state.ordinary_distribution_guard = ordinary_distribution_guard;
                            WorkerSelection::Single { worker }
                        }
                        None => {
                            error!(
                                function = "WorkerSelectionStage::execute",
                                mode = "Regular",
                                model_id = %model_id,
                                "No available workers for model"
                            );
                            return Err(error::model_not_found(model_id));
                        }
                    }
                }
                WorkerSelectionMode::PrefillDecode => {
                    match self.select_pd_pair(model_id, text, tokens, headers) {
                        Some((prefill, decode, runtime_type)) => WorkerSelection::Disaggregated {
                            encode_assignments: None,
                            prefill,
                            decode,
                            runtime_type,
                        },
                        None => {
                            error!(
                                function = "WorkerSelectionStage::execute",
                                mode = "PrefillDecode",
                                model_id = %model_id,
                                "No available PD worker pairs for model"
                            );
                            return Err(error::model_not_found(model_id));
                        }
                    }
                }
                WorkerSelectionMode::EncodePrefillDecode => {
                    let encode_item_hashes = match encode_item_hashes(intermediate) {
                        Ok(hashes) => hashes,
                        Err(err) => {
                            error!(
                                function = "WorkerSelectionStage::execute",
                                error = %err,
                                "Failed to derive encode item routing hashes"
                            );
                            return Err(error::internal_error(
                                "encode_routing_hash_failed",
                                format!("Failed to derive encode routing hashes: {err}"),
                            ));
                        }
                    };
                    match self.select_encode_prefill_decode_workers(
                        model_id,
                        text,
                        tokens,
                        headers,
                        &encode_item_hashes,
                    ) {
                        Some((encode_assignments, prefill, decode, runtime_type)) => {
                            WorkerSelection::Disaggregated {
                                encode_assignments: if encode_assignments.is_empty() {
                                    None
                                } else {
                                    Some(encode_assignments)
                                },
                                prefill,
                                decode,
                                runtime_type,
                            }
                        }
                        None => {
                            error!(
                                function = "WorkerSelectionStage::execute",
                                mode = "EncodePrefillDecode",
                                model_id = %model_id,
                                "No available encode/prefill/decode worker set for model"
                            );
                            return Err(error::model_not_found(model_id));
                        }
                    }
                }
            }
        };

        // Reject an unsupported (backend, modality) combination now that the
        // runtime is known, before request building fetches/preprocesses media
        // only to fail deep in assembly. The prefill leg builds the request in
        // disaggregated mode, so its runtime is the one that must support the
        // request's modalities.
        if let Some(intermediate) = intermediate {
            if let Err(err) = multimodal::ensure_backend_supports_modalities(
                selection_runtime(&workers),
                intermediate,
            ) {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    format!("{err}"),
                ));
            }
        }

        ctx.state.workers = Some(workers);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!("WorkerSelectionStage({:?})", self.mode)
    }
}

/// Runtime of the leg that builds the generate request: the sole worker in
/// regular mode, the prefill worker in disaggregated (PD/EPD) mode.
fn selection_runtime(workers: &WorkerSelection) -> RuntimeType {
    match workers {
        WorkerSelection::Single { worker } => worker.metadata().spec.runtime_type,
        WorkerSelection::Disaggregated { runtime_type, .. } => *runtime_type,
    }
}

fn worker_is_available_for_request(worker: &dyn Worker, headers: Option<&HeaderMap>) -> bool {
    worker.is_available() && worker_url_is_allowed(headers, worker.url())
}

impl WorkerSelectionStage {
    fn select_distribution_seed(
        &self,
        ctx: &RequestContext,
        pending: &PendingDistributionSeed,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Option<(Arc<dyn Worker>, DistributionSeedDispatchGuard)> {
        let controller = ctx.components.adaptive_admission.as_ref()?;
        let workers: Vec<_> = self
            .worker_registry
            .get_workers_filtered(
                Some(model_id),
                Some(WorkerType::Regular),
                Some(ConnectionMode::Grpc),
                None,
                false,
            )
            .into_iter()
            // Header-excluded authoritative owners must remain visible to the
            // cache-policy owner ceiling. SelectWorkerInfo applies the same
            // headers only to actual target eligibility below.
            .filter(|worker| worker.is_available())
            .collect();
        if workers.len() < 2 {
            return None;
        }

        let info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            hash_ring: self.worker_registry.get_hash_ring(model_id),
            max_output_tokens: None,
            reserve_work: false,
            leg: WorkerLeg::Single,
        };
        let meta = ctx.input.tenant_request_meta.as_ref()?;
        let proof = meta.extension::<SchedulerAdmissionProof>()?;
        let claimed = proof
            .try_claim(
                meta.tenant_key(),
                meta.request_charge_id(),
                model_id,
                &pending.partition,
            )
            .ok()?;

        // Claim scheduler authority before mutating policy cooldown or target
        // capacity. The short controller lock then orders this seed selection
        // against ordinary cache-aware selection without crossing an await.
        let selection_guard = controller.lock_distribution_selection();
        let targets = controller.distribution_headroom_snapshot(&pending.partition, model_id);
        if targets.is_empty() {
            return None;
        }
        let capacity: Vec<_> = targets
            .iter()
            .map(|target| SeedWorkerHeadroom {
                worker_url: Arc::from(target.worker_url()),
                worker_revision: target.worker_revision(),
                issuable_slots: target.issuable_slots(),
            })
            .collect();
        let plan = self
            .policy_registry
            .owner_pressure_dispatch_plan(model_id, &workers, &info, &capacity)?;
        let target = targets.iter().find(|target| {
            target.worker_url() == plan.target_worker_url()
                && target.worker_revision() == plan.target_worker_revision()
        })?;
        let selected = workers.into_iter().find(|worker| {
            worker.url() == plan.target_worker_url()
                && worker.revision() == plan.target_worker_revision()
                && worker_is_available_for_request(worker.as_ref(), headers)
        })?;
        if !target.matches_worker(&selected) {
            return None;
        }
        let lease = controller.try_acquire_distribution_headroom(&selection_guard, target)?;
        if !lease.verify() {
            return None;
        }

        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            if plan.expands_ownership() {
                "cache_aware_distribution_seed"
            } else {
                "cache_aware_owner_rebalance"
            },
        );
        Some((
            selected,
            DistributionSeedDispatchGuard {
                headroom: lease,
                _scheduler_proof: claimed,
                policy_plan: plan,
                retry_after_secs: pending.retry_after_secs,
            },
        ))
    }

    #[expect(
        clippy::result_large_err,
        reason = "the pipeline contract returns an Axum Response on local rejection"
    )]
    fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        distribution_scope: Option<(&Arc<AdaptiveAdmissionController>, &str)>,
        cache_credit_chat_eligible: bool,
    ) -> Result<Option<SingleWorkerSelection>, Response> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Distribution-enabled cache-aware routing must see header-excluded
        // authoritative owners so its per-prefix claim can serialize any
        // replacement. The policy itself enforces header eligibility. Active
        // seed targets remain visible for ownership matching but are appended
        // to the policy-only exclusion header until their request-lifetime
        // headroom lease ends. This covers early/deeper KV store events and the
        // routing-key override path without hiding the provisional owner.
        // Other models retain the legacy prefilter.
        let distribution_cache_aware = distribution_scope.and_then(|(controller, partition)| {
            policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
                .map(|cache_aware| (controller, partition, cache_aware))
        });
        let active_targets =
            distribution_cache_aware.map_or_else(Vec::new, |(controller, partition, _)| {
                controller.active_distribution_targets(partition, model_id)
            });
        let distribution_protections = distribution_cache_aware.map_or_else(
            DistributionProtectionSnapshot::default,
            |(_, _, cache_aware)| cache_aware.distribution_protection_snapshot(model_id, tokens),
        );
        let serialize_distribution =
            !active_targets.is_empty() || !distribution_protections.is_empty();
        let policy_headers = if active_targets.is_empty() {
            None
        } else {
            let mut merged = headers.cloned().unwrap_or_default();
            let targeted_worker = merged
                .get("x-smg-target-worker-url")
                .and_then(|value| value.to_str().ok());
            if targeted_worker.is_some_and(|targeted| {
                active_targets
                    .iter()
                    .any(|(url, _)| url.as_ref() == targeted)
            }) {
                return Err(rejection_response(1));
            }
            let mut excluded = merged
                .get("x-smg-excluded-worker-urls")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string();
            for (url, _) in &active_targets {
                if !excluded.is_empty() {
                    excluded.push(',');
                }
                excluded.push_str(url);
            }
            let Ok(excluded) = HeaderValue::from_str(&excluded) else {
                return Err(rejection_response(1));
            };
            merged.insert("x-smg-excluded-worker-urls", excluded);
            Some(merged)
        };
        let effective_headers = policy_headers.as_ref().or(headers);
        let info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers: effective_headers,
            hash_ring,
            max_output_tokens: None,
            reserve_work: true,
            leg: WorkerLeg::Single,
        };
        let available: Vec<Arc<dyn Worker>> = workers
            .into_iter()
            .filter(|worker| {
                if serialize_distribution {
                    worker.is_available()
                } else {
                    worker_is_available_for_request(worker.as_ref(), headers)
                }
            })
            .collect();

        if available.is_empty() {
            return if serialize_distribution {
                Err(rejection_response(1))
            } else {
                Ok(None)
            };
        }

        // Select and reserve prompt/output work atomically. The returned guard
        // releases the reservation when the request finishes or any later
        // gRPC pipeline stage fails.
        let selection = if serialize_distribution {
            let Some((_, _, cache_aware)) = distribution_cache_aware else {
                return Err(rejection_response(1));
            };
            match cache_aware.select_worker_decision_with_distribution_state(
                &available,
                &info,
                &active_targets,
                &distribution_protections,
            ) {
                CacheAwareSelection::Selected(index) => {
                    Some((index, policy.reservation_cost(&info)))
                }
                CacheAwareSelection::Blocked => return Err(rejection_response(1)),
                CacheAwareSelection::Unavailable => return Err(rejection_response(1)),
            }
        } else {
            if cache_credit_chat_eligible {
                self.policy_registry
                    .select_worker_with_reservation_cache_credit_chat(
                        &policy, &available, &info, model_id,
                    )
            } else {
                self.policy_registry
                    .select_worker_with_reservation(&policy, &available, &info)
            }
        };
        let Some((idx, reservation_cost)) = selection else {
            return Ok(None);
        };
        let Some(selected) = available.get(idx).cloned() else {
            return Ok(None);
        };
        let reservation = reservation_cost
            .map(|cost| PolicyReservation::new(policy.clone(), selected.url().to_string(), cost));
        if !worker_url_is_allowed(headers, selected.url()) {
            return Err(rejection_response(1));
        }
        // Cache hashing and ordinary policy selection stay outside the global
        // controller lock. Only this final atomic snapshot check plus exact
        // predispatch claim is serialized against the rare seed path.
        let distribution_selection_guard = distribution_cache_aware
            .map(|(controller, _, _)| controller.lock_distribution_selection());
        if let Some((controller, partition)) = distribution_scope {
            let protection_snapshot_unchanged =
                distribution_cache_aware.is_some_and(|(_, _, cache_aware)| {
                    cache_aware.distribution_protection_snapshot(model_id, tokens)
                        == distribution_protections
                });
            let current_targets = controller.active_distribution_targets(partition, model_id);
            let snapshot_unchanged = current_targets.len() == active_targets.len()
                && current_targets.iter().all(|(url, revision)| {
                    active_targets
                        .iter()
                        .any(|(snapshot_url, snapshot_revision)| {
                            snapshot_url == url && snapshot_revision == revision
                        })
                });
            if !protection_snapshot_unchanged || !snapshot_unchanged {
                return Err(rejection_response(1));
            }
            if controller.distribution_target_is_active(
                partition,
                model_id,
                selected.url(),
                selected.revision(),
            ) {
                return Err(rejection_response(1));
            }
        }
        let ordinary_distribution_guard =
            if let (Some((controller, partition, _)), Some(selection_guard)) = (
                distribution_cache_aware,
                distribution_selection_guard.as_ref(),
            ) {
                Some(
                    controller
                        .try_reserve_ordinary_distribution_target(
                            selection_guard,
                            partition,
                            model_id,
                            &selected,
                        )
                        .ok_or_else(|| rejection_response(1))?,
                )
            } else {
                None
            };

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            policy.name(),
        );

        Ok(Some((selected, reservation, ordinary_distribution_guard)))
    }

    fn select_pd_pair(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Option<PdWorkerPair> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            None,
            Some(ConnectionMode::Grpc), // Match any gRPC worker
            None,                       // any runtime type
            false,
        );

        let (all_prefill, all_decode): (Vec<_>, Vec<_>) =
            all_workers
                .into_iter()
                .fold((Vec::new(), Vec::new()), |mut acc, w| {
                    if worker_is_available_for_request(w.as_ref(), headers) {
                        match w.metadata().spec.worker_type {
                            WorkerType::Prefill => acc.0.push(w),
                            WorkerType::Decode => acc.1.push(w),
                            WorkerType::Regular => {}
                            // Encode-prefill-decode selection is handled in select_encode_prefill_decode_workers;
                            // the PD pair fold ignores encode workers.
                            WorkerType::Encode => {}
                        }
                    }
                    acc
                });

        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }

        if all_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Determine the runtime type from prefill workers.
        // All workers in a PD pair must use the same runtime.
        let first_runtime = all_prefill.first()?.metadata().spec.runtime_type;

        // Check for mixed runtimes in both prefill and decode pools
        let prefill_mixed = all_prefill
            .iter()
            .skip(1)
            .any(|w| w.metadata().spec.runtime_type != first_runtime);
        let decode_mixed = all_decode
            .iter()
            .any(|w| w.metadata().spec.runtime_type != first_runtime);

        if prefill_mixed || decode_mixed {
            warn!(
                "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
                prefill_mixed,
                decode_mixed,
                first_runtime
            );
        }

        let target_runtime = first_runtime;

        // Filter both pools to the target runtime
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if available_prefill.is_empty() || available_decode.is_empty() {
            warn!("No available PD pair for runtime {:?}", target_runtime);
            return None;
        }

        // Select using policies
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Prefill and decode are separate pools; tag each leg so the routing-key
        // override keys its sticky map per leg (a key sticks independently).
        let mut info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            hash_ring,
            max_output_tokens: None,
            reserve_work: false,
            leg: WorkerLeg::Prefill,
        };
        let prefill_idx = self
            .policy_registry
            .select_worker(&policy, &available_prefill, &info)?;
        info.leg = WorkerLeg::Decode;
        let decode_idx = self
            .policy_registry
            .select_worker(&policy, &available_decode, &info)?;

        let model = model_id;
        let policy_name = policy.name();

        // Record worker selection metrics for both prefill and decode
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model,
            policy_name,
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model,
            policy_name,
        );

        Some((
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
            target_runtime,
        ))
    }

    /// Select per-item encode workers + a prefill/decode pair for EPD routing.
    ///
    /// Mirrors `select_pd_pair` but also assigns each multimodal item to an
    /// encode worker. prefill+decode are selected as a normal PD pair. All pools
    /// are filtered to a runtime shared by the selected encode/prefill/decode
    /// legs.
    fn select_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        encode_item_hashes: &[Vec<u8>],
    ) -> Option<EncodePrefillDecodeWorkerSelection> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            None,
            Some(ConnectionMode::Grpc), // Match any gRPC worker
            None,                       // any runtime type
            false,
        );

        let (all_encode, all_prefill, all_decode): (Vec<_>, Vec<_>, Vec<_>) = all_workers
            .into_iter()
            .fold((Vec::new(), Vec::new(), Vec::new()), |mut acc, w| {
                if worker_is_available_for_request(w.as_ref(), headers) {
                    match w.metadata().spec.worker_type {
                        WorkerType::Encode => acc.0.push(w),
                        WorkerType::Prefill => acc.1.push(w),
                        WorkerType::Decode => acc.2.push(w),
                        WorkerType::Regular => {}
                    }
                }
                acc
            });

        let needs_encode = !encode_item_hashes.is_empty();
        if needs_encode && all_encode.is_empty() {
            warn!("No available encode workers");
            return None;
        }
        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }
        if all_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Disaggregated legs must share a runtime. Pick a runtime that has at
        // least one available worker in every required EPD pool instead of
        // blindly using the first prefill runtime.
        let Some(target_runtime) = all_prefill
            .iter()
            .map(|w| w.metadata().spec.runtime_type)
            .find(|runtime| {
                // The current EPD multimodal encoder adapter is TokenSpeed-
                // specific. Do not select a shared SGLang/vLLM runtime only to
                // reject it later during request building.
                (!needs_encode || *runtime == RuntimeType::TokenSpeed)
                    && all_decode
                        .iter()
                        .any(|w| w.metadata().spec.runtime_type == *runtime)
                    && (!needs_encode
                        || all_encode
                            .iter()
                            .any(|w| w.metadata().spec.runtime_type == *runtime))
            })
        else {
            warn!("No available encode/prefill/decode worker set with a shared runtime");
            return None;
        };

        let mixed = all_prefill
            .iter()
            .chain(all_decode.iter())
            .any(|w| w.metadata().spec.runtime_type != target_runtime)
            || (needs_encode
                && all_encode
                    .iter()
                    .any(|w| w.metadata().spec.runtime_type != target_runtime));
        if mixed {
            warn!(
                "Mixed runtime types in encode/prefill/decode workers. Using {:?}.",
                target_runtime
            );
        }

        // Filter all three pools to the target runtime
        let available_encode: Vec<_> = all_encode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if (needs_encode && available_encode.is_empty())
            || available_prefill.is_empty()
            || available_decode.is_empty()
        {
            warn!(
                "No available encode/prefill/decode worker set for runtime {:?}",
                target_runtime
            );
            return None;
        }

        // Select encode, prefill, and decode via their per-role policies. Encode
        // defaults to consistent hashing over each item's content hash; prefill
        // and decode fall back to the main policy when unset.
        let encode_policy = self.policy_registry.get_encode_policy();
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let mut info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            hash_ring: hash_ring.clone(),
            max_output_tokens: None,
            reserve_work: false,
            leg: WorkerLeg::Prefill,
        };
        let prefill_idx =
            self.policy_registry
                .select_worker(&prefill_policy, &available_prefill, &info)?;
        info.leg = WorkerLeg::Decode;
        let decode_idx =
            self.policy_registry
                .select_worker(&decode_policy, &available_decode, &info)?;

        let encode_assignments = assign_encode_workers(
            &available_encode,
            encode_item_hashes,
            model_id,
            encode_policy.as_ref(),
            hash_ring.clone(),
        )?;

        // Record worker selection metrics for prefill and decode, each tagged
        // with the policy that picked it. Encode item assignment metrics are
        // recorded in assign_encode_workers.
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            decode_policy.name(),
        );

        Some((
            encode_assignments,
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
            target_runtime,
        ))
    }
}

fn encode_item_hashes(
    intermediate: Option<&multimodal::MultimodalIntermediate>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let Some(intermediate) = intermediate else {
        return Ok(Vec::new());
    };
    multimodal::encode_routing_hashes(intermediate)
}

fn assign_encode_workers(
    encode_workers: &[Arc<dyn Worker>],
    item_hashes: &[Vec<u8>],
    model_id: &str,
    policy: &dyn LoadBalancingPolicy,
    hash_ring: Option<Arc<HashRing>>,
) -> Option<Vec<EncodeWorkerAssignment>> {
    if item_hashes.is_empty() {
        return Some(Vec::new());
    }

    item_hashes
        .iter()
        .enumerate()
        .map(|(item_index, content_hash)| {
            let routing_headers = encode_routing_headers(content_hash);
            let info = SelectWorkerInfo {
                request_text: None,
                tokens: None,
                headers: Some(&routing_headers),
                hash_ring: hash_ring.clone(),
                max_output_tokens: None,
                reserve_work: false,
                leg: WorkerLeg::Single,
            };
            let worker_idx = policy.select_worker(encode_workers, &info)?;
            let worker = encode_workers[worker_idx].clone();
            Metrics::record_worker_selection(
                metrics_labels::WORKER_ENCODE,
                metrics_labels::CONNECTION_GRPC,
                model_id,
                policy.name(),
            );
            Some(EncodeWorkerAssignment { item_index, worker })
        })
        .collect()
}

fn encode_routing_headers(content_hash: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = hex_encode(content_hash);
    if let Ok(value) = HeaderValue::from_str(&key) {
        headers.insert("x-smg-routing-key", value);
    }
    headers
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::http::{header::RETRY_AFTER, StatusCode};
    use kv_index::{
        compute_content_hash, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap,
    };
    use openai_protocol::{
        chat::ChatCompletionRequest,
        generate::GenerateRequest,
        model_card::ModelCard,
        worker::{HealthCheckConfig, SchedulerLoadSnapshot, WorkerLoadResponse},
    };
    use tokio::sync::watch;

    use super::*;
    use crate::{
        config::{
            AdaptiveAdmissionConfig, AdaptiveAdmissionMode, AdaptiveAdmissionStrategy,
            ManualAssignmentMode, PolicyConfig, RoutingKeyOverrideConfig,
        },
        middleware::scheduler::LocalAdaptiveRejection,
        worker::{BasicWorkerBuilder, KvEventMonitor},
    };

    fn ready_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn stage_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("kimi-k3"))
                .label("admission_partition", "k3")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn clean_headroom_load(timestamp: &str) -> WorkerLoadResponse {
        WorkerLoadResponse {
            timestamp: timestamp.to_string(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_total_reqs: 0,
                token_usage: 0.1,
                utilization: 0.1,
                max_running_requests: 38,
                ..Default::default()
            }],
        }
    }

    fn assert_local_adaptive_rejection(result: Result<Option<SingleWorkerSelection>, Response>) {
        let response = match result {
            Err(response) => response,
            Ok(_) => panic!(
                "active seed B must block locally rather than route to sticky B or uncached C"
            ),
        };
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            response
                .extensions()
                .get::<LocalAdaptiveRejection>()
                .is_some(),
            "the rejection must use the scheduler refund marker"
        );
        assert_eq!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn cache_credit_is_limited_to_direct_scalar_streaming_regular_chat() {
        let direct_streaming_chat = RequestType::Chat(Arc::new(ChatCompletionRequest {
            stream: true,
            ..Default::default()
        }));
        assert!(is_cache_credit_chat_eligible(
            WorkerSelectionMode::Regular,
            StreamingDistributionSeedScope::DirectRegularChat,
            1,
            &direct_streaming_chat,
        ));

        for (name, mode, scope, backend_request_count) in [
            (
                "PD mode",
                WorkerSelectionMode::PrefillDecode,
                StreamingDistributionSeedScope::DirectRegularChat,
                1,
            ),
            (
                "indirect chat",
                WorkerSelectionMode::Regular,
                StreamingDistributionSeedScope::Ineligible,
                1,
            ),
            (
                "backend fanout",
                WorkerSelectionMode::Regular,
                StreamingDistributionSeedScope::DirectRegularChat,
                2,
            ),
        ] {
            assert!(
                !is_cache_credit_chat_eligible(
                    mode,
                    scope,
                    backend_request_count,
                    &direct_streaming_chat,
                ),
                "{name} must use legacy routing"
            );
        }

        let non_streaming_chat = RequestType::Chat(Arc::new(ChatCompletionRequest::default()));
        assert!(!is_cache_credit_chat_eligible(
            WorkerSelectionMode::Regular,
            StreamingDistributionSeedScope::DirectRegularChat,
            1,
            &non_streaming_chat,
        ));

        let multi_choice_chat = RequestType::Chat(Arc::new(ChatCompletionRequest {
            stream: true,
            n: Some(2),
            ..Default::default()
        }));
        assert!(!is_cache_credit_chat_eligible(
            WorkerSelectionMode::Regular,
            StreamingDistributionSeedScope::DirectRegularChat,
            1,
            &multi_choice_chat,
        ));

        let streaming_generate = RequestType::Generate(Arc::new(
            serde_json::from_value::<GenerateRequest>(serde_json::json!({
                "model": "kimi-k3",
                "text": "hello",
                "stream": true,
            }))
            .unwrap(),
        ));
        assert!(!is_cache_credit_chat_eligible(
            WorkerSelectionMode::Regular,
            StreamingDistributionSeedScope::DirectRegularChat,
            1,
            &streaming_generate,
        ));
    }

    #[test]
    fn trusted_worker_url_constraints_filter_before_policy_selection() {
        let public = ready_worker("grpc://public:30000");
        let reserved = ready_worker("grpc://reserved:30000");

        let mut shared_headers = HeaderMap::new();
        shared_headers.insert(
            "x-smg-excluded-worker-urls",
            reserved.url().parse().unwrap(),
        );
        assert!(worker_is_available_for_request(
            public.as_ref(),
            Some(&shared_headers)
        ));
        assert!(!worker_is_available_for_request(
            reserved.as_ref(),
            Some(&shared_headers)
        ));

        let mut private_headers = HeaderMap::new();
        private_headers.insert("x-smg-target-worker-url", reserved.url().parse().unwrap());
        assert!(!worker_is_available_for_request(
            public.as_ref(),
            Some(&private_headers)
        ));
        assert!(worker_is_available_for_request(
            reserved.as_ref(),
            Some(&private_headers)
        ));
    }

    #[tokio::test]
    async fn active_distribution_seed_blocks_sticky_deeper_owner_after_header_filtering() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker_a = stage_worker("grpc://worker-a:30000");
        let worker_b = stage_worker("grpc://worker-b:30000");
        let worker_c = stage_worker("grpc://worker-c:30000");
        for worker in [&worker_a, &worker_b, &worker_c] {
            worker_registry
                .register(Arc::clone(worker))
                .expect("unique worker registration");
        }

        let policy_registry = Arc::new(PolicyRegistry::with_override(
            PolicyConfig::CacheAware {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 0,
                max_tree_size: 10_000,
                fallback_output_token_estimate: 4096,
                block_size: 4,
                engine_load: false,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                max_cached_owners_per_prefix: 8,
                cache_owner_spill_cooldown_secs: 5,
            },
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::MinLoad,
                ..Default::default()
            },
        ));

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.set_block_size("kimi-k3", 4);
        let indexer = Arc::new(PositionalIndexer::new(4));
        let owner_a = indexer.intern_worker(worker_a.url()).unwrap();
        let seed_b = indexer.intern_worker(worker_b.url()).unwrap();
        let prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let owner_blocks: Vec<_> = prefix_tokens
            .chunks(4)
            .enumerate()
            .map(|(index, tokens)| StoredBlock {
                seq_hash: SequenceHash(index as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer
            .apply_stored(owner_a, &owner_blocks, None, &mut WorkerBlockMap::default())
            .unwrap();
        monitor
            .indexers
            .insert("kimi-k3".to_string(), Arc::clone(&indexer));
        policy_registry.set_kv_event_monitor(Some(monitor));

        let policy = policy_registry.get_policy_or_default("kimi-k3");
        let mut sticky_headers = HeaderMap::new();
        sticky_headers.insert("x-smg-routing-key", "trajectory-1".parse().unwrap());
        let sticky_info = SelectWorkerInfo {
            tokens: Some(&prefix_tokens),
            headers: Some(&sticky_headers),
            ..Default::default()
        };
        assert_eq!(
            policy_registry.select_worker(&policy, &[Arc::clone(&worker_b)], &sticky_info),
            Some(0),
            "the routing key should first become sticky to worker B"
        );
        assert_eq!(
            policy_registry.select_worker(
                &policy,
                &[
                    Arc::clone(&worker_a),
                    Arc::clone(&worker_b),
                    Arc::clone(&worker_c),
                ],
                &sticky_info,
            ),
            Some(1),
            "the routing-key override should still resolve to worker B"
        );

        let controller = AdaptiveAdmissionController::new(
            AdaptiveAdmissionConfig {
                mode: AdaptiveAdmissionMode::Enforce,
                strategy: AdaptiveAdmissionStrategy::EngineFeedback,
                distribution_headroom_partitions: vec!["k3".to_string()],
                distribution_headroom_partition_seed_cap: 1,
                ..Default::default()
            },
            Arc::clone(&worker_registry),
        );
        let loads = HashMap::from([
            (
                worker_a.url().to_string(),
                clean_headroom_load("stage-worker-a"),
            ),
            (
                worker_b.url().to_string(),
                clean_headroom_load("stage-worker-b"),
            ),
            (
                worker_c.url().to_string(),
                clean_headroom_load("stage-worker-c"),
            ),
        ]);
        let (_load_tx, load_rx) = watch::channel(loads);
        controller.start_load_updates(load_rx);
        let target_b = controller
            .distribution_headroom_snapshot("k3", "kimi-k3")
            .into_iter()
            .find(|target| target.worker_url() == worker_b.url())
            .expect("worker B should expose clean distribution headroom");
        let seed_lease = {
            let selection = controller.lock_distribution_selection();
            controller
                .try_acquire_distribution_headroom(&selection, &target_b)
                .expect("worker B seed lease")
        };
        assert!(seed_lease.verify());

        let request_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut request_headers = sticky_headers;
        request_headers.insert(
            "x-smg-excluded-worker-urls",
            worker_a.url().parse().unwrap(),
        );
        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::Regular,
        );
        assert_local_adaptive_rejection(stage.select_single_worker(
            "kimi-k3",
            None,
            Some(&request_tokens),
            Some(&request_headers),
            Some((&controller, "k3")),
            false,
        ));

        // The seed's store event can make B a deeper authoritative owner while
        // its request is still in flight. The stage must continue to reject,
        // even though the routing key is sticky to B and A is header-excluded.
        let seed_blocks: Vec<_> = request_tokens
            .chunks(4)
            .enumerate()
            .map(|(index, tokens)| StoredBlock {
                seq_hash: SequenceHash(index as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer
            .apply_stored(seed_b, &seed_blocks, None, &mut WorkerBlockMap::default())
            .unwrap();

        assert_local_adaptive_rejection(stage.select_single_worker(
            "kimi-k3",
            None,
            Some(&request_tokens),
            Some(&request_headers),
            Some((&controller, "k3")),
            false,
        ));
    }
}
