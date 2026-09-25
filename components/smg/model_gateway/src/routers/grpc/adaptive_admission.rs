//! Adaptive admission shared by the gRPC and HTTP serving paths.
//!
//! The existing priority scheduler remains the infrastructure safety layer.
//! The original strategy predicts output-token work. The engine-feedback
//! strategy instead learns each partition's useful running-concurrency knee
//! from live throughput, probes just above it, and backs off on engine queue or
//! KV pressure. Shadow mode exercises either state machine without delaying or
//! rejecting traffic.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::{Mutex, MutexGuard};
use tokio::sync::watch;

use crate::{
    config::{AdaptiveAdmissionConfig, AdaptiveAdmissionMode, AdaptiveAdmissionStrategy},
    middleware::scheduler::state::AdaptiveCapacityProvider,
    observability::metrics::intern_string,
    worker::{registry::WorkerId, Worker, WorkerRegistry},
};

const ADMISSION_PARTITION_LABEL: &str = "admission_partition";
const DISTRIBUTION_HEADROOM_MAX_TELEMETRY_AGE: Duration = Duration::from_secs(30);

const PREDICTIONS_TOTAL: &str = "smg_adaptive_admission_predictions_total";
const PREDICTED_OUTPUT_TOKENS: &str = "smg_adaptive_admission_predicted_output_tokens";
const OBSERVED_OUTPUT_TOKENS: &str = "smg_adaptive_admission_observed_output_tokens";
const ABSOLUTE_ERROR_TOKENS: &str = "smg_adaptive_admission_absolute_error_tokens";
const DECISIONS_TOTAL: &str = "smg_adaptive_admission_decisions_total";
const OUTSTANDING_TOKENS: &str = "smg_adaptive_admission_outstanding_tokens";
const ROUTER_OUTSTANDING_TOKENS: &str = "smg_adaptive_admission_router_outstanding_tokens";
const ENGINE_ESTIMATED_TOKENS: &str = "smg_adaptive_admission_engine_estimated_tokens";
const WORK_BUDGET_TOKENS: &str = "smg_adaptive_admission_work_budget_tokens";
const DRAIN_SECONDS: &str = "smg_adaptive_admission_predicted_drain_seconds";
const LOAD_COVERAGE: &str = "smg_adaptive_admission_load_coverage";
const GEN_THROUGHPUT: &str = "smg_adaptive_admission_generation_tokens_per_second";
const LEARNED_CAPACITY: &str = "smg_adaptive_admission_learned_capacity_tokens_per_second";
const ENGINE_RUNNING: &str = "smg_adaptive_admission_engine_running_requests";
const ENGINE_WAITING: &str = "smg_adaptive_admission_engine_waiting_requests";
const ENGINE_WAITING_TOKENS: &str = "smg_adaptive_admission_engine_waiting_uncached_tokens";
const ENGINE_TOKEN_USAGE: &str = "smg_adaptive_admission_engine_max_token_usage";
const ENGINE_MEAN_TOKEN_USAGE: &str = "smg_adaptive_admission_engine_mean_token_usage";
const ENGINE_MAX_RUNNING: &str = "smg_adaptive_admission_engine_max_running_requests";
const ENGINE_MAX_RUNNING_COVERAGE: &str =
    "smg_adaptive_admission_engine_max_running_requests_coverage";
const ROUTER_OUTSTANDING_REQUESTS: &str = "smg_adaptive_admission_router_outstanding_requests";
const FEEDBACK_RUNNING_LIMIT: &str = "smg_adaptive_admission_feedback_running_limit";
const FEEDBACK_KNEE_PER_REPLICA: &str = "smg_adaptive_admission_feedback_knee_requests_per_replica";
const SEGMENTS: &str = "smg_adaptive_admission_estimator_segments";

pub(crate) const FLAG_MULTIPLE_COMPLETIONS: u16 = 1 << 0;
pub(crate) const FLAG_TOOLS: u16 = 1 << 1;
pub(crate) const FLAG_STRUCTURED_OUTPUT: u16 = 1 << 2;
pub(crate) const FLAG_REASONING: u16 = 1 << 3;
pub(crate) const FLAG_STREAMING: u16 = 1 << 4;

pub(crate) fn describe_metrics() {
    describe_counter!(
        PREDICTIONS_TOTAL,
        "Adaptive output-token predictions by model and fallback level"
    );
    describe_histogram!(
        PREDICTED_OUTPUT_TOKENS,
        "Predicted completion tokens per adaptive-admission request"
    );
    describe_histogram!(
        OBSERVED_OUTPUT_TOKENS,
        "Observed completion tokens for requests learned by adaptive admission"
    );
    describe_histogram!(
        ABSOLUTE_ERROR_TOKENS,
        "Absolute adaptive output-token prediction error"
    );
    describe_counter!(
        DECISIONS_TOTAL,
        "Adaptive token-work admission decisions, including shadow decisions"
    );
    describe_gauge!(
        OUTSTANDING_TOKENS,
        "Effective outstanding output-token estimate used for adaptive admission"
    );
    describe_gauge!(
        ROUTER_OUTSTANDING_TOKENS,
        "Predicted output tokens reserved by this router process"
    );
    describe_gauge!(
        ENGINE_ESTIMATED_TOKENS,
        "Estimated output tokens already running or waiting on engines"
    );
    describe_gauge!(
        WORK_BUDGET_TOKENS,
        "Live output-token work budget derived from engine throughput and the configured horizon"
    );
    describe_gauge!(
        DRAIN_SECONDS,
        "Predicted seconds to drain outstanding output work at current engine throughput"
    );
    describe_gauge!(
        LOAD_COVERAGE,
        "Fraction of healthy replicas with fresh engine-load telemetry"
    );
    describe_gauge!(
        GEN_THROUGHPUT,
        "Aggregate generation throughput reported by engines in an admission partition"
    );
    describe_gauge!(
        LEARNED_CAPACITY,
        "Recent decayed-peak generation capacity learned from engine telemetry"
    );
    describe_gauge!(
        ENGINE_RUNNING,
        "Engine-reported running requests in an admission partition"
    );
    describe_gauge!(
        ENGINE_WAITING,
        "Engine-reported waiting requests in an admission partition"
    );
    describe_gauge!(
        ENGINE_WAITING_TOKENS,
        "Engine-reported waiting uncached tokens in an admission partition"
    );
    describe_gauge!(
        ENGINE_TOKEN_USAGE,
        "Maximum engine-reported token usage in an admission partition"
    );
    describe_gauge!(
        ENGINE_MEAN_TOKEN_USAGE,
        "Mean engine-reported token usage in an admission partition"
    );
    describe_gauge!(
        ENGINE_MAX_RUNNING,
        "Sum of engine-reported maximum running requests in an admission partition"
    );
    describe_gauge!(
        ENGINE_MAX_RUNNING_COVERAGE,
        "Fraction of healthy replicas contributing a maximum-running-requests ceiling"
    );
    describe_gauge!(
        ROUTER_OUTSTANDING_REQUESTS,
        "Requests currently tracked by this router process"
    );
    describe_gauge!(
        FEEDBACK_RUNNING_LIMIT,
        "Dynamic request limit selected by engine-feedback admission"
    );
    describe_gauge!(
        FEEDBACK_KNEE_PER_REPLICA,
        "Learned running requests per replica at the throughput knee"
    );
    describe_gauge!(
        SEGMENTS,
        "Current bounded in-memory output estimator segment count"
    );
}

#[derive(Debug, Clone)]
pub(crate) struct PredictionFeatures {
    pub model: String,
    pub user: String,
    pub workload_type: String,
    pub endpoint: &'static str,
    pub prompt_tokens: u32,
    /// Total upper bound after multiplying per-completion limits by request
    /// multiplicity. `None` means the client did not supply an upper bound.
    pub max_output_tokens: Option<u32>,
    pub generation_flags: u16,
}

impl PredictionFeatures {
    fn prompt_bucket(&self) -> u8 {
        if self.prompt_tokens == 0 {
            0
        } else {
            (u32::BITS - self.prompt_tokens.leading_zeros()) as u8
        }
    }

    fn output_limit_bucket(&self) -> u8 {
        self.max_output_tokens.map_or(0, |tokens| {
            if tokens == 0 {
                0
            } else {
                (u32::BITS - tokens.leading_zeros()) as u8
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PredictionSource {
    ColdStart,
    Model,
    UserOrWorkload,
    UserWorkload,
    Full,
}

impl PredictionSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::ColdStart => "cold_start",
            Self::Model => "model",
            Self::UserOrWorkload => "user_or_workload",
            Self::UserWorkload => "user_workload",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone)]
struct Prediction {
    output_tokens: u32,
    model_output_tokens: u32,
    source: PredictionSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SegmentKey {
    Model(String),
    Workload(String, String),
    User(String, String),
    UserWorkload(String, String, String),
    Full {
        model: String,
        user: String,
        workload: String,
        endpoint: &'static str,
        prompt_bucket: u8,
        output_limit_bucket: u8,
        generation_flags: u16,
    },
}

#[derive(Debug, Clone)]
struct DecayedMean {
    weight: f64,
    weighted_sum: f64,
    last_update: Instant,
}

impl DecayedMean {
    fn new(value: f64, now: Instant) -> Self {
        Self {
            weight: 1.0,
            weighted_sum: value,
            last_update: now,
        }
    }

    fn decay_factor(&self, now: Instant, half_life_secs: f64) -> f64 {
        let elapsed = now
            .saturating_duration_since(self.last_update)
            .as_secs_f64();
        2.0_f64.powf(-elapsed / half_life_secs)
    }

    fn effective(&self, now: Instant, half_life_secs: f64) -> (f64, f64) {
        let decay = self.decay_factor(now, half_life_secs);
        (self.weight * decay, self.weighted_sum * decay)
    }

    fn update(&mut self, value: f64, now: Instant, half_life_secs: f64) {
        let decay = self.decay_factor(now, half_life_secs);
        self.weight = self.weight * decay + 1.0;
        self.weighted_sum = self.weighted_sum * decay + value;
        self.last_update = now;
    }
}

#[derive(Debug)]
struct HierarchicalPredictor {
    half_life_secs: f64,
    prior_observations: f64,
    cold_start_output_tokens: u32,
    max_segments: usize,
    segments: HashMap<SegmentKey, DecayedMean>,
}

impl HierarchicalPredictor {
    fn new(config: &AdaptiveAdmissionConfig) -> Self {
        Self {
            half_life_secs: config.estimator_half_life_secs,
            prior_observations: config.prior_observations,
            cold_start_output_tokens: config.cold_start_output_tokens,
            max_segments: config.max_segments,
            segments: HashMap::new(),
        }
    }

    fn keys(features: &PredictionFeatures) -> [SegmentKey; 5] {
        [
            SegmentKey::Model(features.model.clone()),
            SegmentKey::Workload(features.model.clone(), features.workload_type.clone()),
            SegmentKey::User(features.model.clone(), features.user.clone()),
            SegmentKey::UserWorkload(
                features.model.clone(),
                features.user.clone(),
                features.workload_type.clone(),
            ),
            SegmentKey::Full {
                model: features.model.clone(),
                user: features.user.clone(),
                workload: features.workload_type.clone(),
                endpoint: features.endpoint,
                prompt_bucket: features.prompt_bucket(),
                output_limit_bucket: features.output_limit_bucket(),
                generation_flags: features.generation_flags,
            },
        ]
    }

    fn estimate(&self, key: &SegmentKey, now: Instant) -> Option<(f64, f64)> {
        let (weight, sum) = self.segments.get(key)?.effective(now, self.half_life_secs);
        (weight > f64::EPSILON).then_some((sum / weight, weight))
    }

    fn blend(&self, prior: f64, estimate: Option<(f64, f64)>) -> (f64, bool) {
        let Some((mean, weight)) = estimate else {
            return (prior, false);
        };
        let denominator = weight + self.prior_observations;
        if denominator <= f64::EPSILON {
            return (mean, true);
        }
        (
            (mean * weight + prior * self.prior_observations) / denominator,
            true,
        )
    }

    fn predict_at(&self, features: &PredictionFeatures, now: Instant) -> Prediction {
        let [model, workload, user, user_workload, full] = Self::keys(features);
        let cold = f64::from(self.cold_start_output_tokens);
        let (model_prediction, has_model) = self.blend(cold, self.estimate(&model, now));

        let workload_estimate = self.estimate(&workload, now);
        let user_estimate = self.estimate(&user, now);
        let parent = match (workload_estimate, user_estimate) {
            (Some((workload_mean, workload_weight)), Some((user_mean, user_weight))) => {
                let total = workload_weight + user_weight;
                if total <= f64::EPSILON {
                    model_prediction
                } else {
                    (workload_mean * workload_weight + user_mean * user_weight) / total
                }
            }
            (Some((mean, _)), None) | (None, Some((mean, _))) => mean,
            (None, None) => model_prediction,
        };
        let has_user_or_workload = workload_estimate.is_some() || user_estimate.is_some();
        let (user_workload_prediction, has_user_workload) =
            self.blend(parent, self.estimate(&user_workload, now));
        let (full_prediction, has_full) =
            self.blend(user_workload_prediction, self.estimate(&full, now));

        let source = if has_full {
            PredictionSource::Full
        } else if has_user_workload {
            PredictionSource::UserWorkload
        } else if has_user_or_workload {
            PredictionSource::UserOrWorkload
        } else if has_model {
            PredictionSource::Model
        } else {
            PredictionSource::ColdStart
        };
        let mut output_tokens = full_prediction.round().clamp(1.0, f64::from(u32::MAX)) as u32;
        if let Some(maximum) = features.max_output_tokens {
            output_tokens = output_tokens.min(maximum.max(1));
        }
        Prediction {
            output_tokens,
            model_output_tokens: model_prediction.round().clamp(1.0, f64::from(u32::MAX)) as u32,
            source,
        }
    }

    fn observe_at(&mut self, features: &PredictionFeatures, output_tokens: u32, now: Instant) {
        for key in Self::keys(features) {
            self.segments
                .entry(key)
                .and_modify(|mean| {
                    mean.update(f64::from(output_tokens), now, self.half_life_secs);
                })
                .or_insert_with(|| DecayedMean::new(f64::from(output_tokens), now));
        }
        self.evict_oldest();
    }

    fn evict_oldest(&mut self) {
        if self.segments.len() <= self.max_segments {
            return;
        }
        // Evict a batch so a stream of one-off users does not sort the entire
        // table on every completion after reaching the limit.
        let batch = (self.max_segments / 10).max(1);
        let target = self.max_segments.saturating_sub(batch);
        let excess = self.segments.len().saturating_sub(target);
        let mut oldest: Vec<_> = self
            .segments
            .iter()
            .map(|(key, value)| (key.clone(), value.last_update))
            .collect();
        oldest.sort_unstable_by_key(|(_, updated)| *updated);
        for (key, _) in oldest.into_iter().take(excess) {
            self.segments.remove(&key);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PartitionLoad {
    healthy_replicas: u32,
    observed_replicas: u32,
    generation_tokens_per_second: f64,
    learned_capacity_tokens_per_second: f64,
    running_requests: i64,
    waiting_requests: i64,
    waiting_uncached_tokens: i64,
    max_token_usage: f64,
    token_usage_sum: f64,
    max_running_requests: i64,
    max_running_observed_replicas: u32,
}

#[derive(Debug, Clone)]
struct CapacityEstimate {
    per_replica_tokens_per_second: f64,
    last_update: Instant,
}

impl CapacityEstimate {
    fn effective(&self, now: Instant, half_life_secs: f64) -> f64 {
        let elapsed = now
            .saturating_duration_since(self.last_update)
            .as_secs_f64();
        self.per_replica_tokens_per_second * 2.0_f64.powf(-elapsed / half_life_secs)
    }

    fn observe(&mut self, value: f64, now: Instant, half_life_secs: f64) {
        self.per_replica_tokens_per_second = self.effective(now, half_life_secs).max(value);
        self.last_update = now;
    }
}

#[derive(Debug, Clone)]
struct FeedbackEstimate {
    peak_tokens_per_second_per_replica: f64,
    running_requests_per_replica_at_peak: f64,
    /// True after the partition has crossed a live pressure boundary. This
    /// keeps a tiny idle-throughput sample from becoming a restrictive learned
    /// cap when the engine does not publish `max_running_requests`.
    pressure_observed: bool,
    last_update: Instant,
}

impl FeedbackEstimate {
    fn effective_peak(&self, now: Instant, half_life_secs: f64) -> f64 {
        let elapsed = now
            .saturating_duration_since(self.last_update)
            .as_secs_f64();
        self.peak_tokens_per_second_per_replica * 2.0_f64.powf(-elapsed / half_life_secs)
    }

    fn observe(
        &mut self,
        throughput_per_replica: f64,
        running_per_replica: f64,
        now: Instant,
        half_life_secs: f64,
        improvement_ratio: f64,
        under_pressure: bool,
    ) {
        self.pressure_observed |= under_pressure;
        let effective_peak = self.effective_peak(now, half_life_secs);
        let raises_peak =
            throughput_per_replica > effective_peak * (1.0 + improvement_ratio.max(0.0));
        let same_plateau =
            throughput_per_replica >= effective_peak * (1.0 - improvement_ratio.clamp(0.0, 1.0));
        if raises_peak || same_plateau {
            self.peak_tokens_per_second_per_replica = effective_peak.max(throughput_per_replica);
            if raises_peak || running_per_replica < self.running_requests_per_replica_at_peak {
                self.running_requests_per_replica_at_peak = running_per_replica;
            }
            self.last_update = now;
        }
    }
}

impl PartitionLoad {
    fn coverage(&self) -> f64 {
        if self.healthy_replicas == 0 {
            0.0
        } else {
            f64::from(self.observed_replicas) / f64::from(self.healthy_replicas)
        }
    }

    fn mean_token_usage(&self) -> f64 {
        if self.observed_replicas == 0 {
            0.0
        } else {
            self.token_usage_sum / f64::from(self.observed_replicas)
        }
    }

    fn max_running_coverage(&self) -> f64 {
        if self.healthy_replicas == 0 {
            0.0
        } else {
            f64::from(self.max_running_observed_replicas) / f64::from(self.healthy_replicas)
        }
    }

    fn scaled_max_running_requests(&self) -> f64 {
        if self.max_running_observed_replicas == 0 {
            0.0
        } else {
            (self.max_running_requests as f64 * f64::from(self.healthy_replicas)
                / f64::from(self.max_running_observed_replicas))
            .floor()
        }
    }
}

/// Strict per-worker engine snapshot used only to account distribution
/// headroom. Unlike the aggregate admission signal, every DP rank must be
/// present and internally valid before this record exists.
#[derive(Debug, Clone)]
struct DistributionWorkerTelemetry {
    partition: Arc<str>,
    worker_url: Arc<str>,
    worker_id: WorkerId,
    worker_instance: Arc<dyn Worker>,
    worker_revision: u64,
    telemetry_revision: u64,
    source_timestamp: Arc<str>,
    observed_at: Instant,
    running_requests: u64,
    waiting_requests: u64,
    max_running_requests: u64,
    max_pressure: f64,
}

impl DistributionWorkerTelemetry {
    #[expect(
        clippy::too_many_arguments,
        reason = "constructs one immutable telemetry binding from registry and load identities"
    )]
    fn from_load(
        partition: Arc<str>,
        worker_url: Arc<str>,
        worker_id: WorkerId,
        worker_instance: Arc<dyn Worker>,
        worker_revision: u64,
        telemetry_revision: u64,
        observed_at: Instant,
        load: &WorkerLoadResponse,
    ) -> Option<Self> {
        let source_timestamp = load.timestamp.trim();
        if source_timestamp.is_empty() {
            return None;
        }
        let rank_count = usize::try_from(load.dp_rank_count).ok()?;
        if rank_count == 0 || load.loads.len() != rank_count {
            return None;
        }

        let mut ranks = HashSet::with_capacity(rank_count);
        let mut running_requests = 0_u64;
        let mut waiting_requests = 0_u64;
        let mut max_running_requests = 0_u64;
        let mut max_pressure = 0.0_f64;
        for rank in &load.loads {
            let dp_rank = usize::try_from(rank.dp_rank).ok()?;
            if dp_rank >= rank_count || !ranks.insert(dp_rank) {
                return None;
            }
            if rank.num_running_reqs < 0
                || rank.num_waiting_reqs < 0
                || rank.num_waiting_uncached_tokens < 0
                || rank.num_total_reqs < 0
                || rank.num_used_tokens < 0
                || rank.max_total_num_tokens < 0
                || rank.max_running_requests <= 0
            {
                return None;
            }
            if !rank.token_usage.is_finite()
                || !(0.0..=1.0).contains(&rank.token_usage)
                || !rank.utilization.is_finite()
                || !(0.0..=1.0).contains(&rank.utilization)
            {
                return None;
            }
            if rank.num_total_reqs != rank.num_running_reqs.checked_add(rank.num_waiting_reqs)? {
                return None;
            }
            running_requests = running_requests.checked_add(rank.num_running_reqs as u64)?;
            waiting_requests = waiting_requests.checked_add(rank.num_waiting_reqs as u64)?;
            max_running_requests =
                max_running_requests.checked_add(rank.max_running_requests as u64)?;
            max_pressure = max_pressure.max(rank.token_usage.max(rank.utilization));
        }
        if ranks.len() != rank_count || max_running_requests == 0 {
            return None;
        }

        Some(Self {
            partition,
            worker_url,
            worker_id,
            worker_instance,
            worker_revision,
            telemetry_revision,
            source_timestamp: Arc::from(source_timestamp),
            observed_at,
            running_requests,
            waiting_requests,
            max_running_requests,
            max_pressure,
        })
    }

    fn is_fresh_at(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.observed_at) <= DISTRIBUTION_HEADROOM_MAX_TELEMETRY_AGE
    }

    fn is_clean(&self, max_pressure: f64) -> bool {
        self.waiting_requests == 0 && self.max_pressure < max_pressure
    }

    fn occupied(&self, worker_load: usize) -> u64 {
        let engine_occupied = self.running_requests.saturating_add(self.waiting_requests);
        engine_occupied.max(u64::try_from(worker_load).unwrap_or(u64::MAX))
    }

    fn issuable_slots(&self, worker_load: usize, active_target_leases: u64) -> u16 {
        self.max_running_requests
            .saturating_sub(self.occupied(worker_load))
            .saturating_sub(active_target_leases)
            .min(u64::from(u16::MAX)) as u16
    }
}

#[derive(Debug, Clone)]
struct DistributionHeadroomBinding {
    partition: Arc<str>,
    model: Arc<str>,
    worker_url: Arc<str>,
    worker_id: WorkerId,
    worker_instance: Arc<dyn Worker>,
    worker_revision: u64,
    telemetry_revision: u64,
}

impl PartialEq for DistributionHeadroomBinding {
    fn eq(&self, other: &Self) -> bool {
        self.partition == other.partition
            && self.model == other.model
            && self.worker_url == other.worker_url
            && self.worker_id == other.worker_id
            && Arc::ptr_eq(&self.worker_instance, &other.worker_instance)
            && self.worker_revision == other.worker_revision
            && self.telemetry_revision == other.telemetry_revision
    }
}

impl Eq for DistributionHeadroomBinding {}

/// Read-only, unforgeable snapshot of one worker's currently issuable
/// distribution headroom. Acquisition revalidates every private binding.
#[derive(Debug, Clone)]
pub(crate) struct DistributionHeadroomTarget {
    binding: DistributionHeadroomBinding,
    issuable_slots: u16,
}

impl DistributionHeadroomTarget {
    pub(crate) fn worker_url(&self) -> &str {
        &self.binding.worker_url
    }

    pub(crate) fn worker_revision(&self) -> u64 {
        self.binding.worker_revision
    }

    pub(crate) fn issuable_slots(&self) -> u16 {
        self.issuable_slots
    }

    pub(crate) fn matches_worker(&self, worker: &Arc<dyn Worker>) -> bool {
        self.binding.worker_url.as_ref() == worker.url()
            && self.binding.worker_revision == worker.revision()
            && Arc::ptr_eq(&self.binding.worker_instance, worker)
    }
}

/// Serializes the synchronous ordinary and distribution-seed selection
/// critical sections. It is deliberately short-lived and never crosses an
/// await point.
pub(crate) struct DistributionSelectionGuard<'a> {
    controller: &'a AdaptiveAdmissionController,
    _guard: MutexGuard<'a, ()>,
}

#[derive(Debug)]
struct OrdinaryPredispatchBinding {
    partition: Arc<str>,
    model: Arc<str>,
    worker_id: WorkerId,
    worker_instance: Arc<dyn Worker>,
    worker_revision: u64,
    claims: u64,
}

/// Exact ordinary-route claim held from worker selection until
/// `WorkerLoadGuard` increments the selected worker's live load.
#[derive(Debug)]
pub(crate) struct OrdinaryDistributionDispatchGuard {
    controller: Weak<AdaptiveAdmissionController>,
    partition: Arc<str>,
    model: Arc<str>,
    worker_url: Arc<str>,
    worker_id: WorkerId,
    worker_instance: Arc<dyn Worker>,
    worker_revision: u64,
}

impl Drop for OrdinaryDistributionDispatchGuard {
    fn drop(&mut self) {
        if let Some(controller) = self.controller.upgrade() {
            controller.release_ordinary_distribution_target(self);
        }
    }
}

/// One process-local reservation of a clean worker's engine slot. The lease
/// is non-cloneable and releases itself on every early-return path.
#[derive(Debug)]
pub(crate) struct DistributionHeadroomLease {
    controller: Weak<AdaptiveAdmissionController>,
    binding: Option<DistributionHeadroomBinding>,
}

impl DistributionHeadroomLease {
    /// Revalidate the exact worker, registry revision, telemetry revision,
    /// pressure, and reserved slot immediately before dispatch.
    pub(crate) fn verify(&self) -> bool {
        let Some(binding) = &self.binding else {
            return false;
        };
        self.controller
            .upgrade()
            .is_some_and(|controller| controller.verify_distribution_headroom(binding))
    }
}

impl Drop for DistributionHeadroomLease {
    fn drop(&mut self) {
        let Some(binding) = self.binding.take() else {
            return;
        };
        if let Some(controller) = self.controller.upgrade() {
            controller.release_distribution_headroom(&binding);
        }
    }
}

#[derive(Debug, Default)]
struct WorkState {
    outstanding_tokens: HashMap<String, u64>,
    outstanding_requests: HashMap<String, u64>,
    loads: HashMap<String, PartitionLoad>,
    capacities: HashMap<String, CapacityEstimate>,
    feedback_estimates: HashMap<String, FeedbackEstimate>,
    distribution_telemetry: HashMap<String, DistributionWorkerTelemetry>,
    active_distribution_leases: HashMap<String, DistributionHeadroomBinding>,
    ordinary_predispatch_targets: HashMap<String, OrdinaryPredispatchBinding>,
}

#[derive(Debug, Clone, Copy)]
struct EngineFeedbackConstraints {
    telemetry_usable: bool,
    running_limit: Option<f64>,
    pressure_reason: Option<&'static str>,
}

fn engine_feedback_constraints(
    config: &AdaptiveAdmissionConfig,
    load: &PartitionLoad,
    feedback_estimate: Option<&FeedbackEstimate>,
) -> EngineFeedbackConstraints {
    let telemetry_usable = load.coverage() >= config.min_load_coverage;
    let engine_limit = if load.max_running_coverage() >= config.min_load_coverage {
        Some(load.scaled_max_running_requests())
    } else {
        None
    };
    let learned_limit = feedback_estimate
        .filter(|estimate| engine_limit.is_some() || estimate.pressure_observed)
        .map(|estimate| {
            (estimate.running_requests_per_replica_at_peak * f64::from(load.healthy_replicas))
                .ceil()
                + f64::from(
                    config
                        .feedback_probe_requests_per_healthy_replica
                        .saturating_mul(load.healthy_replicas),
                )
        });
    let running_limit = match (learned_limit, engine_limit) {
        (Some(learned), Some(engine)) => Some(learned.max(1.0).min(engine)),
        (Some(learned), None) => Some(learned.max(1.0)),
        (None, engine) => engine,
    };
    let waiting_limit = i64::from(config.feedback_max_waiting_requests_per_healthy_replica)
        * i64::from(load.observed_replicas);
    let pressure_reason = if load.mean_token_usage() >= config.feedback_max_token_usage {
        Some("token_pressure")
    } else if load.waiting_requests > waiting_limit {
        Some("engine_waiting")
    } else {
        None
    };
    EngineFeedbackConstraints {
        telemetry_usable,
        running_limit,
        pressure_reason,
    }
}

#[derive(Debug)]
struct PredictionSampleState {
    skip_per_model: u64,
    limit_per_model: u64,
    seen_per_model: HashMap<String, u64>,
}

impl PredictionSampleState {
    fn from_env() -> Option<Self> {
        let limit_per_model =
            std::env::var("SMG_ADAPTIVE_ADMISSION_PREDICTION_SAMPLE_LIMIT_PER_MODEL")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
        if limit_per_model == 0 {
            return None;
        }
        let skip_per_model =
            std::env::var("SMG_ADAPTIVE_ADMISSION_PREDICTION_SAMPLE_SKIP_PER_MODEL")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
        Some(Self {
            skip_per_model,
            limit_per_model,
            seen_per_model: HashMap::new(),
        })
    }

    fn should_record(&mut self, model: &str) -> bool {
        let seen = self.seen_per_model.entry(model.to_string()).or_default();
        let record = *seen >= self.skip_per_model
            && *seen < self.skip_per_model.saturating_add(self.limit_per_model);
        *seen = seen.saturating_add(1);
        record
    }
}

#[derive(Debug)]
pub(crate) struct AdaptiveAdmissionController {
    config: AdaptiveAdmissionConfig,
    predictor: Mutex<HierarchicalPredictor>,
    work: Mutex<WorkState>,
    prediction_samples: Option<Mutex<PredictionSampleState>>,
    registry: Arc<WorkerRegistry>,
    capacity_revision: watch::Sender<u64>,
    telemetry_revision: AtomicU64,
    distribution_selection: Mutex<()>,
}

impl AdaptiveAdmissionController {
    pub(crate) fn new(config: AdaptiveAdmissionConfig, registry: Arc<WorkerRegistry>) -> Arc<Self> {
        let (capacity_revision, _) = watch::channel(0_u64);
        Arc::new(Self {
            predictor: Mutex::new(HierarchicalPredictor::new(&config)),
            config,
            work: Mutex::new(WorkState::default()),
            prediction_samples: PredictionSampleState::from_env().map(Mutex::new),
            registry,
            capacity_revision,
            telemetry_revision: AtomicU64::new(0),
            distribution_selection: Mutex::new(()),
        })
    }

    pub(crate) fn lock_distribution_selection(&self) -> DistributionSelectionGuard<'_> {
        DistributionSelectionGuard {
            controller: self,
            _guard: self.distribution_selection.lock(),
        }
    }

    pub(crate) fn mode(&self) -> AdaptiveAdmissionMode {
        self.config.mode
    }

    pub(crate) fn start_load_updates(
        self: &Arc<Self>,
        mut loads: watch::Receiver<HashMap<String, WorkerLoadResponse>>,
    ) {
        self.update_loads(&loads.borrow());
        let controller = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "controller task holds only a weak reference and exits with the gateway"
        )]
        tokio::spawn(async move {
            loop {
                if loads.changed().await.is_err() {
                    break;
                }
                let Some(controller) = controller.upgrade() else {
                    break;
                };
                controller.update_loads(&loads.borrow());
            }
        });
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        let observed_at = Instant::now();
        let telemetry_revision = self
            .telemetry_revision
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut partitions: HashMap<String, PartitionLoad> = HashMap::new();
        let mut distribution_telemetry = HashMap::new();
        for worker in self
            .registry
            .get_all()
            .into_iter()
            .filter(|w| w.is_healthy())
        {
            let partition = worker
                .metadata()
                .spec
                .labels
                .get(ADMISSION_PARTITION_LABEL)
                .map(String::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| worker.model_id())
                .to_string();
            let aggregate = partitions.entry(partition.clone()).or_default();
            aggregate.healthy_replicas = aggregate.healthy_replicas.saturating_add(1);
            let Some(load) = loads.get(worker.url()) else {
                continue;
            };
            if worker.is_available() {
                if let Some(worker_id) = self.registry.get_id_by_url(worker.url()) {
                    if self
                        .registry
                        .get(&worker_id)
                        .is_some_and(|current| Arc::ptr_eq(&current, &worker))
                    {
                        if let Some(telemetry) = DistributionWorkerTelemetry::from_load(
                            Arc::from(partition.as_str()),
                            Arc::from(worker.url()),
                            worker_id,
                            Arc::clone(&worker),
                            worker.revision(),
                            telemetry_revision,
                            observed_at,
                            load,
                        ) {
                            distribution_telemetry.insert(worker.url().to_string(), telemetry);
                        }
                    }
                }
            }
            aggregate.observed_replicas = aggregate.observed_replicas.saturating_add(1);
            aggregate.generation_tokens_per_second += load.total_gen_throughput().max(0.0);
            aggregate.running_requests += load
                .loads
                .iter()
                .map(|rank| i64::from(rank.num_running_reqs.max(0)))
                .sum::<i64>();
            aggregate.waiting_requests += load
                .loads
                .iter()
                .map(|rank| i64::from(rank.num_waiting_reqs.max(0)))
                .sum::<i64>();
            aggregate.waiting_uncached_tokens += load.total_waiting_uncached_tokens().max(0);
            let token_usage = load.effective_token_usage().clamp(0.0, 1.0);
            aggregate.token_usage_sum += token_usage;
            aggregate.max_token_usage = aggregate.max_token_usage.max(token_usage);
            let reported_max_running = load
                .loads
                .iter()
                .map(|rank| i64::from(rank.max_running_requests.max(0)))
                .sum::<i64>();
            let max_running_requests = if reported_max_running > 0 {
                reported_max_running
            } else {
                worker.max_running_requests().map_or(0, i64::from)
            };
            if max_running_requests > 0 {
                aggregate.max_running_requests += max_running_requests;
                aggregate.max_running_observed_replicas =
                    aggregate.max_running_observed_replicas.saturating_add(1);
            }
        }

        let now = observed_at;
        let mut work = self.work.lock();
        // The WorkerMonitor watch snapshot merges independently polled model
        // groups. An unrelated fast poll therefore republishes unchanged K3
        // entries. Preserve the original observation time and binding revision
        // when the backend's per-response timestamp did not advance, so those
        // republishes cannot keep stale distribution telemetry fresh.
        for (url, telemetry) in &mut distribution_telemetry {
            if let Some(previous) = work.distribution_telemetry.get(url) {
                if telemetry.source_timestamp == previous.source_timestamp
                    && telemetry.worker_id == previous.worker_id
                    && Arc::ptr_eq(&telemetry.worker_instance, &previous.worker_instance)
                    && telemetry.worker_revision == previous.worker_revision
                {
                    telemetry.observed_at = previous.observed_at;
                    telemetry.telemetry_revision = previous.telemetry_revision;
                }
            }
        }
        work.capacities
            .retain(|partition, _| partitions.contains_key(partition));
        work.feedback_estimates
            .retain(|partition, _| partitions.contains_key(partition));
        for (partition, load) in &mut partitions {
            if load.observed_replicas > 0 && load.generation_tokens_per_second > 0.0 {
                let per_replica =
                    load.generation_tokens_per_second / f64::from(load.observed_replicas);
                work.capacities
                    .entry(partition.clone())
                    .and_modify(|capacity| {
                        capacity.observe(per_replica, now, self.config.estimator_half_life_secs);
                    })
                    .or_insert(CapacityEstimate {
                        per_replica_tokens_per_second: per_replica,
                        last_update: now,
                    });
                if load.running_requests > 0 {
                    let running_per_replica =
                        load.running_requests as f64 / f64::from(load.observed_replicas);
                    let waiting_limit = i64::from(
                        self.config
                            .feedback_max_waiting_requests_per_healthy_replica,
                    ) * i64::from(load.observed_replicas);
                    let under_pressure = load.mean_token_usage()
                        >= self.config.feedback_max_token_usage
                        || load.waiting_requests > waiting_limit;
                    work.feedback_estimates
                        .entry(partition.clone())
                        .and_modify(|estimate| {
                            estimate.observe(
                                per_replica,
                                running_per_replica,
                                now,
                                self.config.estimator_half_life_secs,
                                self.config.feedback_throughput_improvement_ratio,
                                under_pressure,
                            );
                        })
                        .or_insert(FeedbackEstimate {
                            peak_tokens_per_second_per_replica: per_replica,
                            running_requests_per_replica_at_peak: running_per_replica,
                            pressure_observed: under_pressure,
                            last_update: now,
                        });
                }
            }
            if let Some(capacity) = work.capacities.get(partition) {
                load.learned_capacity_tokens_per_second = capacity
                    .effective(now, self.config.estimator_half_life_secs)
                    * f64::from(load.healthy_replicas);
            }
        }
        work.loads = partitions;
        work.distribution_telemetry = distribution_telemetry;
        for (partition, load) in &work.loads {
            let partition_label = intern_string(partition);
            gauge!(LOAD_COVERAGE, "partition" => Arc::clone(&partition_label)).set(load.coverage());
            gauge!(GEN_THROUGHPUT, "partition" => Arc::clone(&partition_label))
                .set(load.generation_tokens_per_second);
            gauge!(LEARNED_CAPACITY, "partition" => Arc::clone(&partition_label))
                .set(load.learned_capacity_tokens_per_second);
            gauge!(ENGINE_RUNNING, "partition" => Arc::clone(&partition_label))
                .set(load.running_requests as f64);
            gauge!(ENGINE_WAITING, "partition" => Arc::clone(&partition_label))
                .set(load.waiting_requests as f64);
            gauge!(ENGINE_WAITING_TOKENS, "partition" => Arc::clone(&partition_label))
                .set(load.waiting_uncached_tokens as f64);
            gauge!(ENGINE_TOKEN_USAGE, "partition" => Arc::clone(&partition_label))
                .set(load.max_token_usage);
            gauge!(ENGINE_MEAN_TOKEN_USAGE, "partition" => Arc::clone(&partition_label))
                .set(load.mean_token_usage());
            gauge!(ENGINE_MAX_RUNNING, "partition" => Arc::clone(&partition_label))
                .set(load.max_running_requests as f64);
            gauge!(ENGINE_MAX_RUNNING_COVERAGE, "partition" => Arc::clone(&partition_label))
                .set(load.max_running_coverage());
            gauge!(FEEDBACK_KNEE_PER_REPLICA, "partition" => partition_label).set(
                work.feedback_estimates
                    .get(partition)
                    .map_or(0.0, |estimate| {
                        estimate.running_requests_per_replica_at_peak
                    }),
            );
        }
        drop(work);
        self.capacity_revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
    }

    pub(crate) fn distribution_headroom_enabled(&self, partition: &str) -> bool {
        self.config.mode == AdaptiveAdmissionMode::Enforce
            && self.config.strategy == AdaptiveAdmissionStrategy::EngineFeedback
            && self.config.distribution_headroom_partition_seed_cap == 1
            && self
                .config
                .distribution_headroom_partitions
                .iter()
                .any(|allowed| allowed == partition)
    }

    /// Exact targets currently reserved by in-flight distribution seeds for
    /// this model/partition. Ordinary routing excludes these workers until the
    /// request-lifetime lease is released, even if KV store events arrive
    /// before the seed response completes.
    pub(crate) fn active_distribution_targets(
        &self,
        partition: &str,
        model: &str,
    ) -> Vec<(Arc<str>, u64)> {
        let work = self.work.lock();
        work.active_distribution_leases
            .values()
            .filter(|binding| {
                binding.partition.as_ref() == partition && binding.model.as_ref() == model
            })
            .map(|binding| (Arc::clone(&binding.worker_url), binding.worker_revision))
            .collect()
    }

    pub(crate) fn distribution_target_is_active(
        &self,
        partition: &str,
        model: &str,
        worker_url: &str,
        worker_revision: u64,
    ) -> bool {
        self.work
            .lock()
            .active_distribution_leases
            .get(worker_url)
            .is_some_and(|binding| {
                binding.partition.as_ref() == partition
                    && binding.model.as_ref() == model
                    && binding.worker_revision == worker_revision
            })
    }

    pub(crate) fn try_reserve_ordinary_distribution_target(
        self: &Arc<Self>,
        selection: &DistributionSelectionGuard<'_>,
        partition: &str,
        model: &str,
        worker: &Arc<dyn Worker>,
    ) -> Option<OrdinaryDistributionDispatchGuard> {
        if !std::ptr::eq(self.as_ref(), selection.controller)
            || !self.distribution_headroom_enabled(partition)
            || self.registry.resolve_model_alias(model).is_some()
        {
            return None;
        }
        let worker_id = self.registry.get_id_by_url(worker.url())?;
        let current = self.registry.get(&worker_id)?;
        let worker_partition = current
            .metadata()
            .spec
            .labels
            .get(ADMISSION_PARTITION_LABEL)
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| current.model_id());
        if !Arc::ptr_eq(&current, worker)
            || current.revision() != worker.revision()
            || !current.is_available()
            || worker_partition != partition
            || !self
                .registry
                .get_by_model(model)
                .iter()
                .any(|candidate| Arc::ptr_eq(candidate, worker))
        {
            return None;
        }

        let mut work = self.work.lock();
        if work.active_distribution_leases.contains_key(worker.url()) {
            return None;
        }
        if let Some(binding) = work.ordinary_predispatch_targets.get_mut(worker.url()) {
            if binding.partition.as_ref() != partition
                || binding.model.as_ref() != model
                || binding.worker_id != worker_id
                || !Arc::ptr_eq(&binding.worker_instance, worker)
                || binding.worker_revision != worker.revision()
            {
                return None;
            }
            binding.claims = binding.claims.checked_add(1)?;
        } else {
            work.ordinary_predispatch_targets.insert(
                worker.url().to_string(),
                OrdinaryPredispatchBinding {
                    partition: Arc::from(partition),
                    model: Arc::from(model),
                    worker_id: worker_id.clone(),
                    worker_instance: Arc::clone(worker),
                    worker_revision: worker.revision(),
                    claims: 1,
                },
            );
        }
        Some(OrdinaryDistributionDispatchGuard {
            controller: Arc::downgrade(self),
            partition: Arc::from(partition),
            model: Arc::from(model),
            worker_url: Arc::from(worker.url()),
            worker_id,
            worker_instance: Arc::clone(worker),
            worker_revision: worker.revision(),
        })
    }

    fn release_ordinary_distribution_target(&self, guard: &OrdinaryDistributionDispatchGuard) {
        let mut work = self.work.lock();
        let remove = {
            let Some(binding) = work
                .ordinary_predispatch_targets
                .get_mut(guard.worker_url.as_ref())
            else {
                return;
            };
            if binding.partition != guard.partition
                || binding.model != guard.model
                || binding.worker_id != guard.worker_id
                || !Arc::ptr_eq(&binding.worker_instance, &guard.worker_instance)
                || binding.worker_revision != guard.worker_revision
                || binding.claims == 0
            {
                return;
            }
            binding.claims -= 1;
            binding.claims == 0
        };
        if remove {
            work.ordinary_predispatch_targets
                .remove(guard.worker_url.as_ref());
        }
    }

    fn current_model_worker(
        &self,
        binding: &DistributionHeadroomBinding,
    ) -> Option<Arc<dyn Worker>> {
        // A caller must supply the canonical model selected at request entry.
        // Aliases are not accepted as an independently reusable scope.
        if self.registry.resolve_model_alias(&binding.model).is_some() {
            return None;
        }
        if self.registry.get_id_by_url(&binding.worker_url).as_ref() != Some(&binding.worker_id) {
            return None;
        }
        let worker = self.registry.get(&binding.worker_id)?;
        if worker.url() != binding.worker_url.as_ref()
            || worker.revision() != binding.worker_revision
            || !worker.is_available()
            || !Arc::ptr_eq(&worker, &binding.worker_instance)
        {
            return None;
        }
        self.registry
            .get_by_model(&binding.model)
            .iter()
            .any(|candidate| {
                Arc::ptr_eq(candidate, &binding.worker_instance)
                    && candidate.url() == binding.worker_url.as_ref()
                    && candidate.revision() == binding.worker_revision
                    && candidate.is_available()
            })
            .then_some(worker)
    }

    fn telemetry_matches_binding(
        telemetry: &DistributionWorkerTelemetry,
        binding: &DistributionHeadroomBinding,
        now: Instant,
    ) -> bool {
        telemetry.partition == binding.partition
            && telemetry.worker_url == binding.worker_url
            && telemetry.worker_id == binding.worker_id
            && Arc::ptr_eq(&telemetry.worker_instance, &binding.worker_instance)
            && telemetry.worker_revision == binding.worker_revision
            && telemetry.telemetry_revision == binding.telemetry_revision
            && telemetry.is_fresh_at(now)
    }

    fn active_partition_leases(work: &WorkState, partition: &str) -> u64 {
        u64::try_from(
            work.active_distribution_leases
                .values()
                .filter(|binding| binding.partition.as_ref() == partition)
                .count(),
        )
        .unwrap_or(u64::MAX)
    }

    /// Return a complete strictly validated worker-capacity snapshot for one
    /// canonical model and exact admission partition. Hot, full, and currently
    /// leased workers remain visible with zero issuable slots so routing can
    /// prove owner saturation without mistaking an omitted owner for capacity.
    /// Missing, malformed, stale, or unavailable workers are omitted and thus
    /// fail closed. Targets are advisory; acquisition revalidates every
    /// private binding under the lease lock.
    pub(crate) fn distribution_headroom_snapshot(
        &self,
        partition: &str,
        model: &str,
    ) -> Vec<DistributionHeadroomTarget> {
        if !self.distribution_headroom_enabled(partition)
            || self.registry.resolve_model_alias(model).is_some()
        {
            return Vec::new();
        }
        let workers = self.registry.get_by_model(model);
        if workers.is_empty() {
            return Vec::new();
        }

        let now = Instant::now();
        let work = self.work.lock();
        let partition_has_capacity = Self::active_partition_leases(&work, partition)
            < u64::from(self.config.distribution_headroom_partition_seed_cap);
        let mut targets = Vec::new();
        for worker in workers.iter().filter(|worker| worker.is_available()) {
            let Some(telemetry) = work.distribution_telemetry.get(worker.url()) else {
                continue;
            };
            let Some(worker_id) = self.registry.get_id_by_url(worker.url()) else {
                continue;
            };
            if worker_id != telemetry.worker_id || !Arc::ptr_eq(worker, &telemetry.worker_instance)
            {
                continue;
            }
            let binding = DistributionHeadroomBinding {
                partition: Arc::from(partition),
                model: Arc::from(model),
                worker_url: Arc::from(worker.url()),
                worker_id,
                worker_instance: Arc::clone(worker),
                worker_revision: worker.revision(),
                telemetry_revision: telemetry.telemetry_revision,
            };
            if !Self::telemetry_matches_binding(telemetry, &binding, now) {
                continue;
            }
            let active_target_leases =
                u64::from(work.active_distribution_leases.contains_key(worker.url()));
            let ordinary_predispatch = work
                .ordinary_predispatch_targets
                .get(worker.url())
                .map_or(0, |binding| binding.claims);
            let raw_issuable = telemetry.issuable_slots(
                worker.load(),
                active_target_leases.saturating_add(ordinary_predispatch),
            );
            let issuable_slots = if partition_has_capacity
                && active_target_leases == 0
                && ordinary_predispatch == 0
                && telemetry.is_clean(self.config.feedback_max_token_usage)
            {
                raw_issuable
            } else {
                0
            };
            targets.push(DistributionHeadroomTarget {
                binding,
                issuable_slots,
            });
        }
        targets.sort_unstable_by(|left, right| left.worker_url().cmp(right.worker_url()));
        targets
    }

    /// Acquire one exact clean-worker slot. This is intentionally independent
    /// of routing policy and scheduler authorization; callers must layer those
    /// separate decisions around this process-local capacity reservation.
    pub(crate) fn try_acquire_distribution_headroom(
        self: &Arc<Self>,
        selection: &DistributionSelectionGuard<'_>,
        target: &DistributionHeadroomTarget,
    ) -> Option<DistributionHeadroomLease> {
        let binding = &target.binding;
        if !std::ptr::eq(self.as_ref(), selection.controller)
            || target.issuable_slots == 0
            || !self.distribution_headroom_enabled(&binding.partition)
        {
            return None;
        }

        let worker = self.current_model_worker(binding)?;
        let now = Instant::now();
        let mut work = self.work.lock();
        if Self::active_partition_leases(&work, &binding.partition)
            >= u64::from(self.config.distribution_headroom_partition_seed_cap)
            || work
                .active_distribution_leases
                .contains_key(binding.worker_url.as_ref())
            || work
                .ordinary_predispatch_targets
                .contains_key(binding.worker_url.as_ref())
        {
            return None;
        }
        let telemetry = work
            .distribution_telemetry
            .get(binding.worker_url.as_ref())?;
        if !Self::telemetry_matches_binding(telemetry, binding, now)
            || !telemetry.is_clean(self.config.feedback_max_token_usage)
            || telemetry.issuable_slots(worker.load(), 0) == 0
        {
            return None;
        }
        // Re-read mutable worker state after all telemetry checks. A later
        // registry or health transition is caught by the mandatory verify.
        if worker.revision() != binding.worker_revision || !worker.is_available() {
            return None;
        }
        work.active_distribution_leases
            .insert(binding.worker_url.to_string(), binding.clone());
        drop(work);
        self.capacity_revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
        if self.current_model_worker(binding).is_none() {
            self.release_distribution_headroom(binding);
            return None;
        }
        Some(DistributionHeadroomLease {
            controller: Arc::downgrade(self),
            binding: Some(binding.clone()),
        })
    }

    fn verify_distribution_headroom(&self, binding: &DistributionHeadroomBinding) -> bool {
        if !self.distribution_headroom_enabled(&binding.partition) {
            return false;
        }
        let Some(worker) = self.current_model_worker(binding) else {
            return false;
        };
        let now = Instant::now();
        let work = self.work.lock();
        if work
            .active_distribution_leases
            .get(binding.worker_url.as_ref())
            != Some(binding)
        {
            return false;
        }
        let active_partition_leases = Self::active_partition_leases(&work, &binding.partition);
        if active_partition_leases == 0
            || active_partition_leases
                > u64::from(self.config.distribution_headroom_partition_seed_cap)
        {
            return false;
        }
        let Some(telemetry) = work.distribution_telemetry.get(binding.worker_url.as_ref()) else {
            return false;
        };
        if !Self::telemetry_matches_binding(telemetry, binding, now)
            || !telemetry.is_clean(self.config.feedback_max_token_usage)
        {
            return false;
        }
        let active_target_leases = 1_u64;
        let occupied = telemetry.occupied(worker.load());
        if telemetry.max_running_requests.saturating_sub(occupied) < active_target_leases {
            return false;
        }
        // Evaluate the specified remaining-headroom formula even when this
        // lease consumes the final slot. Zero remaining slots is still a valid
        // reservation for this already-held lease.
        let _remaining_issuable = telemetry.issuable_slots(worker.load(), active_target_leases);
        drop(work);
        worker.revision() == binding.worker_revision
            && worker.is_available()
            && self.current_model_worker(binding).is_some()
    }

    fn release_distribution_headroom(&self, binding: &DistributionHeadroomBinding) {
        let mut work = self.work.lock();
        let removed = if work
            .active_distribution_leases
            .get(binding.worker_url.as_ref())
            == Some(binding)
        {
            work.active_distribution_leases
                .remove(binding.worker_url.as_ref())
                .is_some()
        } else {
            false
        };
        drop(work);
        if removed {
            self.capacity_revision
                .send_modify(|revision| *revision = revision.wrapping_add(1));
        }
    }

    pub(crate) fn begin(
        self: &Arc<Self>,
        partition: String,
        features: PredictionFeatures,
    ) -> AdaptiveRequestTracker {
        // Comet injects a trusted partition header, but standalone SMG users
        // can send arbitrary headers. Only retain a selector already known
        // from fresh worker telemetry; otherwise fall back to the request's
        // model. This keeps state and metric cardinality bounded.
        let partition = {
            let work = self.work.lock();
            if partition == features.model || work.loads.contains_key(&partition) {
                partition
            } else {
                features.model.clone()
            }
        };
        let now = Instant::now();
        let prediction = if self.config.strategy == AdaptiveAdmissionStrategy::PredictedWork {
            let prediction = self.predictor.lock().predict_at(&features, now);
            let model_label = intern_string(&features.model);
            counter!(
                PREDICTIONS_TOTAL,
                "model" => Arc::clone(&model_label),
                "source" => prediction.source.as_str()
            )
            .increment(1);
            histogram!(PREDICTED_OUTPUT_TOKENS, "model" => model_label)
                .record(f64::from(prediction.output_tokens));
            prediction
        } else {
            // Engine feedback does not predict completion length. Retain a
            // zero reservation only so both strategies share the same tracker
            // lifecycle without adding predictor lock contention.
            Prediction {
                output_tokens: 0,
                model_output_tokens: 0,
                source: PredictionSource::ColdStart,
            }
        };

        let decision = {
            let mut work = self.work.lock();
            let load = work.loads.get(&partition).cloned().unwrap_or_default();
            let feedback_estimate = work.feedback_estimates.get(&partition).cloned();
            let outstanding = work
                .outstanding_tokens
                .entry(partition.clone())
                .or_default();
            let prior_outstanding = *outstanding;
            *outstanding = outstanding.saturating_add(u64::from(prediction.output_tokens));
            let router_outstanding = *outstanding as f64;
            let outstanding_requests = work
                .outstanding_requests
                .entry(partition.clone())
                .or_default();
            *outstanding_requests = outstanding_requests.saturating_add(1);
            let router_outstanding_requests = *outstanding_requests as f64;
            let coverage = load.coverage();
            let engine_request_count = load
                .running_requests
                .saturating_add(load.waiting_requests)
                .max(0) as f64;
            let partition_label = intern_string(&partition);
            gauge!(ROUTER_OUTSTANDING_REQUESTS, "partition" => Arc::clone(&partition_label))
                .set(router_outstanding_requests);

            match self.config.strategy {
                AdaptiveAdmissionStrategy::PredictedWork => {
                    let telemetry_usable = coverage >= self.config.min_load_coverage
                        && load.learned_capacity_tokens_per_second.is_finite()
                        && load.learned_capacity_tokens_per_second > 0.0;
                    let budget = if telemetry_usable {
                        load.learned_capacity_tokens_per_second * self.config.work_horizon_secs
                    } else {
                        f64::INFINITY
                    };
                    let engine_estimated =
                        engine_request_count * f64::from(prediction.model_output_tokens);
                    // `max` avoids counting work both in router reservations
                    // and in a later engine poll. Include the incoming request
                    // because it is not in the engine snapshot yet.
                    let projected = router_outstanding
                        .max(engine_estimated + f64::from(prediction.output_tokens));
                    let drain_seconds = if telemetry_usable {
                        projected / load.learned_capacity_tokens_per_second
                    } else {
                        0.0
                    };
                    let would_admit =
                        !telemetry_usable || projected <= budget || prior_outstanding == 0;
                    gauge!(OUTSTANDING_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(projected);
                    gauge!(ROUTER_OUTSTANDING_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(router_outstanding);
                    gauge!(ENGINE_ESTIMATED_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(engine_estimated);
                    gauge!(WORK_BUDGET_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(if budget.is_finite() { budget } else { 0.0 });
                    gauge!(DRAIN_SECONDS, "partition" => Arc::clone(&partition_label))
                        .set(drain_seconds);
                    gauge!(FEEDBACK_RUNNING_LIMIT, "partition" => partition_label).set(0.0);
                    AdmissionDecision {
                        would_admit,
                        telemetry_usable,
                        reason: if !telemetry_usable {
                            "telemetry_fallback"
                        } else if would_admit {
                            "within_work_budget"
                        } else {
                            "work_budget"
                        },
                        retry_after_secs: if would_admit || !telemetry_usable {
                            0
                        } else {
                            ((projected - budget) / load.learned_capacity_tokens_per_second)
                                .ceil()
                                .clamp(1.0, f64::from(u32::MAX)) as u32
                        },
                    }
                }
                AdaptiveAdmissionStrategy::EngineFeedback => {
                    let constraints = engine_feedback_constraints(
                        &self.config,
                        &load,
                        feedback_estimate.as_ref(),
                    );
                    let projected_requests =
                        router_outstanding_requests.max(engine_request_count + 1.0);
                    let reason = if !constraints.telemetry_usable {
                        "telemetry_fallback"
                    } else if let Some(reason) = constraints.pressure_reason {
                        reason
                    } else if constraints
                        .running_limit
                        .is_some_and(|running_limit| projected_requests > running_limit)
                    {
                        "running_limit"
                    } else {
                        "within_feedback_limit"
                    };
                    let would_admit =
                        matches!(reason, "telemetry_fallback" | "within_feedback_limit");
                    gauge!(FEEDBACK_RUNNING_LIMIT, "partition" => Arc::clone(&partition_label))
                        .set(constraints.running_limit.unwrap_or(0.0));
                    gauge!(OUTSTANDING_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(router_outstanding);
                    gauge!(ROUTER_OUTSTANDING_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(router_outstanding);
                    gauge!(ENGINE_ESTIMATED_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(0.0);
                    gauge!(WORK_BUDGET_TOKENS, "partition" => Arc::clone(&partition_label))
                        .set(0.0);
                    gauge!(DRAIN_SECONDS, "partition" => partition_label).set(0.0);
                    AdmissionDecision {
                        would_admit,
                        telemetry_usable: constraints.telemetry_usable,
                        reason,
                        retry_after_secs: u32::from(!would_admit),
                    }
                }
            }
        };

        let outcome = if !decision.telemetry_usable {
            "telemetry_fallback"
        } else if decision.would_admit {
            "would_admit"
        } else {
            "would_reject"
        };
        counter!(
            DECISIONS_TOTAL,
            "partition" => intern_string(&partition),
            "mode" => match self.config.mode {
                AdaptiveAdmissionMode::Off => "off",
                AdaptiveAdmissionMode::Shadow => "shadow",
                AdaptiveAdmissionMode::Enforce => "enforce",
            },
            "strategy" => match self.config.strategy {
                AdaptiveAdmissionStrategy::PredictedWork => "predicted_work",
                AdaptiveAdmissionStrategy::EngineFeedback => "engine_feedback",
            },
            "reason" => decision.reason,
            "outcome" => outcome
        )
        .increment(1);

        AdaptiveRequestTracker {
            inner: Some(TrackerInner {
                controller: Arc::downgrade(self),
                partition,
                features,
                prediction,
                decision,
                resolved: AtomicBool::new(false),
            }),
        }
    }

    fn finish(&self, inner: &TrackerInner, observed_output_tokens: Option<u32>) {
        {
            let mut work = self.work.lock();
            let outstanding = work
                .outstanding_tokens
                .entry(inner.partition.clone())
                .or_default();
            *outstanding = outstanding.saturating_sub(u64::from(inner.prediction.output_tokens));
            gauge!(OUTSTANDING_TOKENS, "partition" => intern_string(&inner.partition))
                .set(*outstanding as f64);
            let outstanding_requests = work
                .outstanding_requests
                .entry(inner.partition.clone())
                .or_default();
            *outstanding_requests = outstanding_requests.saturating_sub(1);
            gauge!(ROUTER_OUTSTANDING_REQUESTS, "partition" => intern_string(&inner.partition))
                .set(*outstanding_requests as f64);
        }
        let Some(observed) = observed_output_tokens else {
            return;
        };
        if self.config.strategy == AdaptiveAdmissionStrategy::EngineFeedback {
            return;
        }
        self.predictor
            .lock()
            .observe_at(&inner.features, observed, Instant::now());
        let model_label = intern_string(&inner.features.model);
        histogram!(OBSERVED_OUTPUT_TOKENS, "model" => Arc::clone(&model_label))
            .record(f64::from(observed));
        histogram!(ABSOLUTE_ERROR_TOKENS, "model" => model_label)
            .record((f64::from(observed) - f64::from(inner.prediction.output_tokens)).abs());
        gauge!(SEGMENTS).set(self.predictor.lock().segments.len() as f64);

        let Some(samples) = &self.prediction_samples else {
            return;
        };
        if samples.lock().should_record(&inner.features.model) {
            tracing::info!(
                target: "smg::adaptive_admission_prediction_sample",
                model = %inner.features.model,
                predicted_output_tokens = inner.prediction.output_tokens,
                observed_output_tokens = observed,
                prediction_source = inner.prediction.source.as_str(),
                prompt_tokens = inner.features.prompt_tokens,
                output_limit_tokens = inner.features.max_output_tokens.unwrap_or(0),
                generation_flags = inner.features.generation_flags,
                "adaptive admission prediction sample"
            );
        }
    }
}

impl AdaptiveCapacityProvider for AdaptiveAdmissionController {
    fn effective_capacity(&self, partition: &str, static_capacity: u16) -> u16 {
        if self.config.mode != AdaptiveAdmissionMode::Enforce
            || self.config.strategy != AdaptiveAdmissionStrategy::EngineFeedback
        {
            return static_capacity;
        }
        let work = self.work.lock();
        let load = work.loads.get(partition).cloned().unwrap_or_default();
        let constraints = engine_feedback_constraints(
            &self.config,
            &load,
            work.feedback_estimates.get(partition),
        );
        if !constraints.telemetry_usable {
            static_capacity
        } else if constraints.pressure_reason.is_some() {
            0
        } else if let Some(running_limit) = constraints.running_limit {
            (running_limit.floor().clamp(0.0, f64::from(u16::MAX)) as u16).min(static_capacity)
        } else {
            static_capacity
        }
    }

    fn subscribe_capacity_changes(&self) -> watch::Receiver<u64> {
        self.capacity_revision.subscribe()
    }
}

#[derive(Debug, Clone, Copy)]
struct AdmissionDecision {
    would_admit: bool,
    telemetry_usable: bool,
    reason: &'static str,
    retry_after_secs: u32,
}

struct TrackerInner {
    controller: Weak<AdaptiveAdmissionController>,
    partition: String,
    features: PredictionFeatures,
    prediction: Prediction,
    decision: AdmissionDecision,
    resolved: AtomicBool,
}

/// Per-request adaptive-admission state. Dropping an unfinished tracker
/// releases its request reservation. Predicted-work mode also releases its
/// token reservation without teaching the estimator from a partial or failed
/// response.
pub(crate) struct AdaptiveRequestTracker {
    inner: Option<TrackerInner>,
}

impl AdaptiveRequestTracker {
    pub(crate) fn should_reject(&self) -> bool {
        let Some(inner) = &self.inner else {
            return false;
        };
        inner.controller.upgrade().is_some_and(|controller| {
            controller.mode() == AdaptiveAdmissionMode::Enforce
                && inner.decision.telemetry_usable
                && !inner.decision.would_admit
        })
    }

    pub(crate) fn retry_after_secs(&self) -> u32 {
        self.inner
            .as_ref()
            .map_or(0, |inner| inner.decision.retry_after_secs)
    }

    /// Exact admission partition bound when this tracker was created.
    pub(crate) fn partition(&self) -> Option<&str> {
        self.inner.as_ref().map(|inner| inner.partition.as_str())
    }

    /// Machine-stable reason for an enforced, telemetry-backed rejection.
    /// Admitted and telemetry-fallback trackers do not report a rejection.
    pub(crate) fn rejection_reason(&self) -> Option<&'static str> {
        self.inner.as_ref().and_then(|inner| {
            (inner.decision.telemetry_usable && !inner.decision.would_admit)
                .then_some(inner.decision.reason)
        })
    }

    pub(crate) fn complete(mut self, observed_output_tokens: u32) {
        self.resolve(Some(observed_output_tokens));
    }

    fn resolve(&mut self, observed_output_tokens: Option<u32>) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        if inner.resolved.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(controller) = inner.controller.upgrade() {
            controller.finish(&inner, observed_output_tokens);
        }
    }
}

impl Drop for AdaptiveRequestTracker {
    fn drop(&mut self) {
        self.resolve(None);
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, time::Duration};

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, SchedulerLoadSnapshot, WorkerStatus},
    };

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerLoadGuard};

    fn config() -> AdaptiveAdmissionConfig {
        AdaptiveAdmissionConfig {
            mode: AdaptiveAdmissionMode::Shadow,
            strategy: AdaptiveAdmissionStrategy::PredictedWork,
            work_horizon_secs: 10.0,
            estimator_half_life_secs: 60.0,
            prior_observations: 2.0,
            max_segments: 20,
            min_load_coverage: 0.8,
            cold_start_output_tokens: 100,
            feedback_probe_requests_per_healthy_replica: 2,
            feedback_max_waiting_requests_per_healthy_replica: 2,
            feedback_max_token_usage: 0.9,
            feedback_throughput_improvement_ratio: 0.02,
            distribution_headroom_partitions: Vec::new(),
            distribution_headroom_partition_seed_cap: 0,
        }
    }

    fn distribution_config(cap: u32) -> AdaptiveAdmissionConfig {
        AdaptiveAdmissionConfig {
            mode: AdaptiveAdmissionMode::Enforce,
            strategy: AdaptiveAdmissionStrategy::EngineFeedback,
            distribution_headroom_partitions: vec!["k3".to_string()],
            distribution_headroom_partition_seed_cap: cap,
            ..config()
        }
    }

    fn register_headroom_worker(registry: &Arc<WorkerRegistry>, url: &str) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("model").with_alias("model-alias"))
                .label(ADMISSION_PARTITION_LABEL, "k3")
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry
            .register(Arc::clone(&worker))
            .expect("worker registration");
        worker
    }

    fn worker_load(ranks: &[(i32, i32, i32, f64, f64, i32)]) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: i32::try_from(ranks.len()).expect("test rank count"),
            loads: ranks
                .iter()
                .map(
                    |&(dp_rank, running, waiting, token_usage, utilization, maximum)| {
                        SchedulerLoadSnapshot {
                            dp_rank,
                            num_running_reqs: running,
                            num_waiting_reqs: waiting,
                            num_total_reqs: running.saturating_add(waiting),
                            token_usage,
                            utilization,
                            max_running_requests: maximum,
                            ..Default::default()
                        }
                    },
                )
                .collect(),
            ..Default::default()
        }
    }

    fn update_worker_loads(
        controller: &AdaptiveAdmissionController,
        loads: impl IntoIterator<Item = (&'static str, WorkerLoadResponse)>,
    ) {
        static TEST_LOAD_REVISION: AtomicU64 = AtomicU64::new(0);
        controller.update_loads(
            &loads
                .into_iter()
                .map(|(url, mut load)| {
                    if load.timestamp.is_empty() {
                        load.timestamp = format!(
                            "test-{}",
                            TEST_LOAD_REVISION.fetch_add(1, Ordering::Relaxed)
                        );
                    }
                    (url.to_string(), load)
                })
                .collect(),
        );
    }

    fn acquire_distribution_headroom(
        controller: &Arc<AdaptiveAdmissionController>,
        target: &DistributionHeadroomTarget,
    ) -> Option<DistributionHeadroomLease> {
        let selection = controller.lock_distribution_selection();
        controller.try_acquire_distribution_headroom(&selection, target)
    }

    fn features(user: &str, prompt_tokens: u32, maximum: Option<u32>) -> PredictionFeatures {
        PredictionFeatures {
            model: "model".to_string(),
            user: user.to_string(),
            workload_type: "rollout".to_string(),
            endpoint: "chat",
            prompt_tokens,
            max_output_tokens: maximum,
            generation_flags: 0,
        }
    }

    #[test]
    fn prediction_samples_skip_warmup_and_cap_each_model() {
        let mut samples = PredictionSampleState {
            skip_per_model: 2,
            limit_per_model: 2,
            seen_per_model: HashMap::new(),
        };

        assert!(!samples.should_record("model-a"));
        assert!(!samples.should_record("model-a"));
        assert!(samples.should_record("model-a"));
        assert!(samples.should_record("model-a"));
        assert!(!samples.should_record("model-a"));
        assert!(!samples.should_record("model-b"));
        assert!(!samples.should_record("model-b"));
        assert!(samples.should_record("model-b"));
    }

    #[test]
    fn cold_start_is_clamped_by_request_limit() {
        let predictor = HierarchicalPredictor::new(&config());
        let prediction = predictor.predict_at(&features("u", 10, Some(32)), Instant::now());
        assert_eq!(prediction.output_tokens, 32);
        assert_eq!(prediction.source, PredictionSource::ColdStart);
    }

    #[test]
    fn user_history_beats_model_mean_and_decays() {
        let mut predictor = HierarchicalPredictor::new(&config());
        let start = Instant::now();
        let user_a = features("a", 1000, None);
        let user_b = features("b", 1000, None);
        for i in 0..20 {
            let now = start + Duration::from_secs(i);
            predictor.observe_at(&user_a, 20, now);
            predictor.observe_at(&user_b, 200, now);
        }
        let prediction = predictor.predict_at(&user_a, start + Duration::from_secs(21));
        assert!(prediction.output_tokens < 80, "{prediction:?}");
        assert!(matches!(
            prediction.source,
            PredictionSource::UserWorkload | PredictionSource::Full
        ));

        predictor.observe_at(&user_a, 400, start + Duration::from_secs(600));
        let shifted = predictor.predict_at(&user_a, start + Duration::from_secs(601));
        assert!(shifted.output_tokens > prediction.output_tokens);
    }

    #[test]
    fn estimator_state_is_bounded() {
        let mut predictor = HierarchicalPredictor::new(&config());
        let start = Instant::now();
        for i in 0..100 {
            predictor.observe_at(
                &features(&format!("user-{i}"), i + 1, None),
                i + 1,
                start + Duration::from_secs(u64::from(i)),
            );
        }
        assert!(predictor.segments.len() <= predictor.max_segments);
        assert!(predictor.segments.len() >= predictor.max_segments / 2);
    }

    #[test]
    fn learned_capacity_tracks_recent_peak_and_decays() {
        let start = Instant::now();
        let mut capacity = CapacityEstimate {
            per_replica_tokens_per_second: 100.0,
            last_update: start,
        };
        capacity.observe(40.0, start + Duration::from_secs(60), 60.0);
        assert!((capacity.per_replica_tokens_per_second - 50.0).abs() < f64::EPSILON);
        capacity.observe(80.0, start + Duration::from_secs(61), 60.0);
        assert!((capacity.per_replica_tokens_per_second - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_partition_header_falls_back_to_model() {
        let controller =
            AdaptiveAdmissionController::new(config(), Arc::new(WorkerRegistry::new()));
        let tracker = controller.begin("attacker-controlled".to_string(), features("a", 10, None));
        assert_eq!(tracker.inner.as_ref().unwrap().partition, "model");
    }

    #[test]
    fn work_horizon_decision_uses_throughput_not_request_count() {
        let registry = Arc::new(WorkerRegistry::new());
        let controller = AdaptiveAdmissionController::new(config(), registry);
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                generation_tokens_per_second: 10.0,
                learned_capacity_tokens_per_second: 10.0,
                ..PartitionLoad::default()
            },
        );

        let first = controller.begin("model".to_string(), features("a", 10, None));
        assert!(!first.should_reject(), "shadow mode never rejects");
        let second = controller.begin("model".to_string(), features("b", 10, None));
        assert!(!second.inner.as_ref().unwrap().decision.would_admit);
        drop(second);
        drop(first);
        assert_eq!(
            controller.work.lock().outstanding_tokens.get("model"),
            Some(&0)
        );
    }

    #[test]
    fn enforce_rejects_only_after_live_work_budget_is_exhausted() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                generation_tokens_per_second: 10.0,
                learned_capacity_tokens_per_second: 10.0,
                ..PartitionLoad::default()
            },
        );

        let first = controller.begin("model".to_string(), features("a", 10, None));
        assert!(!first.should_reject());
        let second = controller.begin("model".to_string(), features("b", 10, None));
        assert!(second.should_reject());
        assert_eq!(second.retry_after_secs(), 10);
    }

    #[test]
    fn enforce_fails_open_when_engine_load_coverage_is_incomplete() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 2,
                observed_replicas: 1,
                generation_tokens_per_second: 10.0,
                learned_capacity_tokens_per_second: 20.0,
                ..PartitionLoad::default()
            },
        );

        let first = controller.begin("model".to_string(), features("a", 10, None));
        let second = controller.begin("model".to_string(), features("b", 10, None));
        assert!(!first.should_reject());
        assert!(!second.should_reject());
        assert!(!second.inner.as_ref().unwrap().decision.telemetry_usable);
    }

    #[test]
    fn engine_backlog_survives_router_reservation_loss() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.work_horizon_secs = 30.0;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                generation_tokens_per_second: 10.0,
                learned_capacity_tokens_per_second: 10.0,
                running_requests: 3,
                ..PartitionLoad::default()
            },
        );

        let starvation_probe = controller.begin("model".to_string(), features("a", 10, None));
        assert!(!starvation_probe.should_reject());
        let next = controller.begin("model".to_string(), features("b", 10, None));
        assert!(next.should_reject());
    }

    #[test]
    fn engine_feedback_uses_learned_knee_plus_probe_margin() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        let now = Instant::now();
        let mut work = controller.work.lock();
        work.loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                generation_tokens_per_second: 100.0,
                running_requests: 10,
                max_running_requests: 64,
                max_running_observed_replicas: 1,
                max_token_usage: 0.5,
                ..PartitionLoad::default()
            },
        );
        work.feedback_estimates.insert(
            "model".to_string(),
            FeedbackEstimate {
                peak_tokens_per_second_per_replica: 100.0,
                running_requests_per_replica_at_peak: 10.0,
                pressure_observed: false,
                last_update: now,
            },
        );
        drop(work);

        let trackers: Vec<_> = (0..12)
            .map(|i| controller.begin("model".to_string(), features(&format!("u-{i}"), 10, None)))
            .collect();
        assert!(trackers.iter().all(|tracker| !tracker.should_reject()));
        let excess = controller.begin("model".to_string(), features("excess", 10, None));
        assert!(excess.should_reject());
        assert_eq!(excess.retry_after_secs(), 1);
    }

    #[test]
    fn engine_feedback_backs_off_on_waiting_or_token_pressure() {
        for (waiting_requests, token_usage, expected_reason) in
            [(3, 0.5, "engine_waiting"), (0, 0.91, "token_pressure")]
        {
            let mut settings = config();
            settings.mode = AdaptiveAdmissionMode::Enforce;
            settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
            let controller =
                AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
            controller.work.lock().loads.insert(
                "model".to_string(),
                PartitionLoad {
                    healthy_replicas: 1,
                    observed_replicas: 1,
                    generation_tokens_per_second: 100.0,
                    running_requests: 1,
                    waiting_requests,
                    max_running_requests: 64,
                    max_running_observed_replicas: 1,
                    max_token_usage: token_usage,
                    token_usage_sum: token_usage,
                    ..PartitionLoad::default()
                },
            );

            let tracker = controller.begin("model".to_string(), features("u", 10, None));
            assert!(tracker.should_reject());
            assert_eq!(tracker.partition(), Some("model"));
            assert_eq!(tracker.rejection_reason(), Some(expected_reason));
            assert_eq!(
                tracker.inner.as_ref().unwrap().decision.reason,
                expected_reason
            );
        }
    }

    #[test]
    fn engine_feedback_cold_start_is_bounded_by_engine_limit() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                max_running_requests: 2,
                max_running_observed_replicas: 1,
                ..PartitionLoad::default()
            },
        );

        let first = controller.begin("model".to_string(), features("a", 10, None));
        let second = controller.begin("model".to_string(), features("b", 10, None));
        let third = controller.begin("model".to_string(), features("c", 10, None));
        assert!(!first.should_reject());
        assert!(!second.should_reject());
        assert!(third.should_reject());
    }

    #[test]
    fn engine_feedback_scales_only_sufficient_max_running_coverage() {
        let sufficiently_covered = PartitionLoad {
            healthy_replicas: 5,
            observed_replicas: 4,
            max_running_requests: 256,
            max_running_observed_replicas: 4,
            ..PartitionLoad::default()
        };
        assert_eq!(sufficiently_covered.coverage(), 0.8);
        assert_eq!(sufficiently_covered.max_running_coverage(), 0.8);
        assert_eq!(sufficiently_covered.scaled_max_running_requests(), 320.0);

        let insufficiently_covered = PartitionLoad {
            healthy_replicas: 5,
            observed_replicas: 5,
            max_running_requests: 192,
            max_running_observed_replicas: 3,
            ..PartitionLoad::default()
        };
        assert_eq!(insufficiently_covered.coverage(), 1.0);
        assert_eq!(insufficiently_covered.max_running_coverage(), 0.6);
    }

    #[test]
    fn engine_feedback_uses_load_when_max_running_coverage_is_incomplete() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 5,
                observed_replicas: 5,
                max_running_requests: 192,
                max_running_observed_replicas: 3,
                ..PartitionLoad::default()
            },
        );

        let tracker = controller.begin("model".to_string(), features("u", 10, None));
        assert!(!tracker.should_reject());
        assert!(tracker.inner.as_ref().unwrap().decision.telemetry_usable);
        assert_eq!(
            tracker.inner.as_ref().unwrap().decision.reason,
            "within_feedback_limit"
        );
    }

    #[test]
    fn engine_feedback_enforces_pressure_without_max_running_limit() {
        for (waiting_requests, token_usage, expected_reason) in
            [(3, 0.5, "engine_waiting"), (0, 0.91, "token_pressure")]
        {
            let mut settings = config();
            settings.mode = AdaptiveAdmissionMode::Enforce;
            settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
            let controller =
                AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
            controller.work.lock().loads.insert(
                "model".to_string(),
                PartitionLoad {
                    healthy_replicas: 1,
                    observed_replicas: 1,
                    running_requests: 10,
                    waiting_requests,
                    max_token_usage: token_usage,
                    token_usage_sum: token_usage,
                    ..PartitionLoad::default()
                },
            );

            let tracker = controller.begin("model".to_string(), features("u", 10, None));
            assert!(tracker.should_reject());
            assert!(tracker.inner.as_ref().unwrap().decision.telemetry_usable);
            assert_eq!(
                tracker.inner.as_ref().unwrap().decision.reason,
                expected_reason
            );
        }
    }

    #[test]
    fn engine_feedback_ignores_idle_knee_without_max_running_limit() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        let now = Instant::now();
        let mut work = controller.work.lock();
        work.loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 4,
                observed_replicas: 4,
                ..PartitionLoad::default()
            },
        );
        work.feedback_estimates.insert(
            "model".to_string(),
            FeedbackEstimate {
                peak_tokens_per_second_per_replica: 0.001,
                running_requests_per_replica_at_peak: 0.25,
                pressure_observed: false,
                last_update: now,
            },
        );
        drop(work);

        let trackers: Vec<_> = (0..20)
            .map(|i| controller.begin("model".to_string(), features(&format!("u-{i}"), 10, None)))
            .collect();
        assert!(trackers.iter().all(|tracker| !tracker.should_reject()));
    }

    #[test]
    fn engine_feedback_uses_busy_knee_without_max_running_limit() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        let now = Instant::now();
        let mut work = controller.work.lock();
        work.loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                running_requests: 10,
                ..PartitionLoad::default()
            },
        );
        work.feedback_estimates.insert(
            "model".to_string(),
            FeedbackEstimate {
                peak_tokens_per_second_per_replica: 100.0,
                running_requests_per_replica_at_peak: 10.0,
                pressure_observed: true,
                last_update: now,
            },
        );
        drop(work);

        let trackers: Vec<_> = (0..12)
            .map(|i| controller.begin("model".to_string(), features(&format!("u-{i}"), 10, None)))
            .collect();
        assert!(trackers.iter().all(|tracker| !tracker.should_reject()));
        let excess = controller.begin("model".to_string(), features("excess", 10, None));
        assert!(excess.should_reject());
        assert_eq!(
            excess.inner.as_ref().unwrap().decision.reason,
            "running_limit"
        );
    }

    #[test]
    fn capacity_provider_reports_total_ceiling_without_request_double_accounting() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let controller =
            AdaptiveAdmissionController::new(settings, Arc::new(WorkerRegistry::new()));
        controller.work.lock().loads.insert(
            "model".to_string(),
            PartitionLoad {
                healthy_replicas: 1,
                observed_replicas: 1,
                running_requests: 63,
                max_running_requests: 64,
                max_running_observed_replicas: 1,
                token_usage_sum: 0.5,
                max_token_usage: 0.5,
                ..PartitionLoad::default()
            },
        );

        assert_eq!(controller.effective_capacity("model", 100), 64);
        controller
            .work
            .lock()
            .loads
            .get_mut("model")
            .unwrap()
            .token_usage_sum = 0.95;
        assert_eq!(controller.effective_capacity("model", 100), 0);
    }

    #[test]
    fn engine_feedback_aggregate_pressure_remains_mean_across_workers() {
        let mut settings = config();
        settings.mode = AdaptiveAdmissionMode::Enforce;
        settings.strategy = AdaptiveAdmissionStrategy::EngineFeedback;
        let load = PartitionLoad {
            healthy_replicas: 2,
            observed_replicas: 2,
            max_token_usage: 0.95,
            token_usage_sum: 0.95,
            max_running_requests: 128,
            max_running_observed_replicas: 2,
            ..PartitionLoad::default()
        };

        let constraints = engine_feedback_constraints(&settings, &load, None);
        assert_eq!(load.mean_token_usage(), 0.475);
        assert_eq!(constraints.pressure_reason, None);
        assert_eq!(constraints.running_limit, Some(128.0));
    }

    #[test]
    fn capacity_revision_changes_on_load_samples_not_per_request() {
        let controller =
            AdaptiveAdmissionController::new(config(), Arc::new(WorkerRegistry::new()));
        let revision = controller.subscribe_capacity_changes();
        let tracker = controller.begin("model".to_string(), features("a", 10, None));
        assert!(!revision.has_changed().unwrap());
        drop(tracker);
        assert!(!revision.has_changed().unwrap());

        controller.update_loads(&HashMap::new());
        assert!(revision.has_changed().unwrap());
    }

    #[test]
    fn distribution_headroom_exposes_idle_peers_behind_one_hot_worker() {
        const HOT: &str = "grpc://hot:30000";
        const IDLE_A: &str = "grpc://idle-a:30000";
        const IDLE_B: &str = "grpc://idle-b:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, HOT);
        register_headroom_worker(&registry, IDLE_A);
        let idle_b = register_headroom_worker(&registry, IDLE_B);
        for _ in 0..5 {
            idle_b.increment_load();
        }
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(
            &controller,
            [
                (HOT, worker_load(&[(0, 38, 64, 0.89, 0.4, 100)])),
                (IDLE_A, worker_load(&[(0, 0, 0, 0.1, 0.2, 43)])),
                (IDLE_B, worker_load(&[(0, 0, 0, 0.1, 0.2, 43)])),
            ],
        );

        assert_eq!(
            controller.effective_capacity("k3", 512),
            0,
            "routing headroom must not mint untyped scheduler capacity"
        );
        let targets = controller.distribution_headroom_snapshot("k3", "model");
        assert_eq!(targets.len(), 3);
        assert_eq!(targets[0].worker_url(), HOT);
        assert_eq!(targets[0].issuable_slots(), 0);
        assert_eq!(targets[1].worker_url(), IDLE_A);
        assert_eq!(targets[1].issuable_slots(), 43);
        assert_eq!(targets[2].worker_url(), IDLE_B);
        assert_eq!(targets[2].issuable_slots(), 38);
    }

    #[test]
    fn distribution_headroom_uses_max_pressure_across_every_rank() {
        const URL: &str = "grpc://dp2:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, URL);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(
            &controller,
            [(
                URL,
                worker_load(&[(0, 0, 0, 0.1, 0.2, 24), (1, 0, 0, 0.2, 0.95, 24)]),
            )],
        );

        {
            let work = controller.work.lock();
            assert_eq!(
                work.distribution_telemetry.get(URL).unwrap().max_pressure,
                0.95
            );
            assert!(
                (work.loads.get("k3").unwrap().mean_token_usage() - 0.15).abs() < f64::EPSILON,
                "the existing aggregate path must retain mean token usage"
            );
        }
        let snapshot = controller.distribution_headroom_snapshot("k3", "model");
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].issuable_slots(), 0);

        update_worker_loads(
            &controller,
            [(
                URL,
                worker_load(&[(0, 0, 0, 0.1, 0.2, 24), (1, 0, 0, 0.2, 0.8, 24)]),
            )],
        );
        let targets = controller.distribution_headroom_snapshot("k3", "model");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].issuable_slots(), 48);
    }

    #[test]
    fn distribution_headroom_never_raises_aggregate_scheduler_capacity() {
        const HOT: &str = "grpc://floor-hot:30000";
        const IDLE_A: &str = "grpc://floor-idle-a:30000";
        const IDLE_B: &str = "grpc://floor-idle-b:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, HOT);
        register_headroom_worker(&registry, IDLE_A);
        register_headroom_worker(&registry, IDLE_B);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(
            &controller,
            [
                (HOT, worker_load(&[(0, 38, 64, 0.89, 0.4, 100)])),
                (IDLE_A, worker_load(&[(0, 0, 0, 0.1, 0.2, 43)])),
                (IDLE_B, worker_load(&[(0, 0, 0, 0.1, 0.2, 43)])),
            ],
        );

        assert_eq!(controller.effective_capacity("k3", 512), 0);
        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .into_iter()
            .find(|target| target.issuable_slots() > 0)
            .unwrap();
        let lease = acquire_distribution_headroom(&controller, &target).unwrap();
        assert_eq!(
            controller.effective_capacity("k3", 512),
            0,
            "a routing lease must never mint an untyped scheduler slot"
        );
        drop(lease);
        assert_eq!(controller.effective_capacity("k3", 512), 0);
    }

    #[test]
    fn distribution_headroom_does_not_reopen_capacity_with_incomplete_telemetry() {
        const HOT: &str = "grpc://incomplete-hot:30000";
        const MISSING: &str = "grpc://incomplete-missing:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, HOT);
        register_headroom_worker(&registry, MISSING);
        let mut settings = distribution_config(1);
        settings.min_load_coverage = 0.5;
        let controller = AdaptiveAdmissionController::new(settings, registry);
        update_worker_loads(
            &controller,
            [(HOT, worker_load(&[(0, 38, 64, 0.89, 0.4, 100)]))],
        );

        assert_eq!(
            controller.effective_capacity("k3", 512),
            0,
            "missing strict telemetry must not reopen aggregate pressure"
        );
    }

    #[test]
    fn distribution_headroom_does_not_reopen_capacity_when_disabled() {
        const HOT: &str = "grpc://off-hot:30000";
        const IDLE: &str = "grpc://off-idle:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, HOT);
        register_headroom_worker(&registry, IDLE);
        let mut settings = distribution_config(1);
        settings.distribution_headroom_partitions.clear();
        settings.distribution_headroom_partition_seed_cap = 0;
        let controller = AdaptiveAdmissionController::new(settings, registry);
        update_worker_loads(
            &controller,
            [
                (HOT, worker_load(&[(0, 38, 64, 0.89, 0.4, 100)])),
                (IDLE, worker_load(&[(0, 0, 0, 0.1, 0.2, 43)])),
            ],
        );

        assert_eq!(controller.effective_capacity("k3", 512), 0);
    }

    #[test]
    fn distribution_headroom_fails_closed_on_missing_malformed_or_stale_telemetry() {
        const URL: &str = "grpc://strict:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, URL);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);

        controller.update_loads(&HashMap::new());
        assert!(controller
            .distribution_headroom_snapshot("k3", "model")
            .is_empty());

        let mut inconsistent_total = worker_load(&[(0, 1, 1, 0.1, 0.1, 16)]);
        inconsistent_total.loads[0].num_total_reqs = 1;
        let malformed = [
            WorkerLoadResponse {
                dp_rank_count: 2,
                loads: worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]).loads,
                ..Default::default()
            },
            worker_load(&[(0, 0, 0, 0.1, 0.1, 16), (0, 0, 0, 0.1, 0.1, 16)]),
            worker_load(&[(0, -1, 0, 0.1, 0.1, 16)]),
            worker_load(&[(0, 0, 0, 0.1, 0.1, 0)]),
            worker_load(&[(0, 0, 0, f64::NAN, 0.1, 16)]),
            inconsistent_total,
        ];
        for load in malformed {
            update_worker_loads(&controller, [(URL, load)]);
            assert!(controller
                .distribution_headroom_snapshot("k3", "model")
                .is_empty());
        }

        update_worker_loads(
            &controller,
            [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]))],
        );
        controller
            .work
            .lock()
            .distribution_telemetry
            .get_mut(URL)
            .unwrap()
            .observed_at = Instant::now().checked_sub(Duration::from_secs(31)).unwrap();
        assert!(controller
            .distribution_headroom_snapshot("k3", "model")
            .is_empty());

        let mut unchanged = worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]);
        unchanged.timestamp = "fixed-backend-sample".to_string();
        let unchanged_map = HashMap::from([(URL.to_string(), unchanged)]);
        controller.update_loads(&unchanged_map);
        controller
            .work
            .lock()
            .distribution_telemetry
            .get_mut(URL)
            .unwrap()
            .observed_at = Instant::now().checked_sub(Duration::from_secs(31)).unwrap();
        controller.update_loads(&unchanged_map);
        assert!(
            controller
                .distribution_headroom_snapshot("k3", "model")
                .is_empty(),
            "an unrelated watch update must not refresh an unchanged backend sample"
        );
    }

    #[test]
    fn distribution_headroom_allowlist_and_cap_are_default_off_and_exact() {
        const URL: &str = "grpc://off:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, URL);

        for settings in [
            AdaptiveAdmissionConfig {
                mode: AdaptiveAdmissionMode::Enforce,
                strategy: AdaptiveAdmissionStrategy::EngineFeedback,
                ..config()
            },
            AdaptiveAdmissionConfig {
                mode: AdaptiveAdmissionMode::Enforce,
                strategy: AdaptiveAdmissionStrategy::EngineFeedback,
                distribution_headroom_partitions: vec!["k3".to_string()],
                distribution_headroom_partition_seed_cap: 0,
                ..config()
            },
            AdaptiveAdmissionConfig {
                mode: AdaptiveAdmissionMode::Enforce,
                strategy: AdaptiveAdmissionStrategy::EngineFeedback,
                distribution_headroom_partitions: vec!["k3-canary".to_string()],
                distribution_headroom_partition_seed_cap: 1,
                ..config()
            },
        ] {
            let controller = AdaptiveAdmissionController::new(settings, Arc::clone(&registry));
            update_worker_loads(
                &controller,
                [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]))],
            );
            assert!(controller
                .distribution_headroom_snapshot("k3", "model")
                .is_empty());
        }
    }

    #[test]
    fn distribution_headroom_concurrently_enforces_target_and_partition_caps() {
        const A: &str = "grpc://cap-a:30000";
        const B: &str = "grpc://cap-b:30000";
        const C: &str = "grpc://cap-c:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, A);
        register_headroom_worker(&registry, B);
        register_headroom_worker(&registry, C);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(
            &controller,
            [
                (A, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)])),
                (B, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)])),
                (C, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)])),
            ],
        );

        let one_target = controller
            .distribution_headroom_snapshot("k3", "model")
            .into_iter()
            .next()
            .unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let claim_handles: Vec<_> = (0..8)
            .map(|_| {
                let controller = Arc::clone(&controller);
                let target = one_target.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    acquire_distribution_headroom(&controller, &target)
                })
            })
            .collect();
        let claims: Vec<_> = claim_handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(claims.len(), 1);
        drop(claims);

        let targets = controller.distribution_headroom_snapshot("k3", "model");
        let barrier = Arc::new(Barrier::new(targets.len()));
        let lease_handles: Vec<_> = targets
            .into_iter()
            .map(|target| {
                let controller = Arc::clone(&controller);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    acquire_distribution_headroom(&controller, &target)
                })
            })
            .collect();
        let leases: Vec<_> = lease_handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(leases.len(), 1);
        assert_eq!(
            AdaptiveAdmissionController::active_partition_leases(&controller.work.lock(), "k3"),
            1
        );
    }

    #[test]
    fn distribution_headroom_lease_drop_refunds_capacity() {
        const URL: &str = "grpc://refund:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, URL);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(&controller, [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 1)]))]);

        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        let lease = acquire_distribution_headroom(&controller, &target).unwrap();
        assert!(lease.verify());
        assert_eq!(
            controller.active_distribution_targets("k3", "model"),
            vec![(Arc::from(URL), target.worker_revision())]
        );
        assert!(controller.distribution_target_is_active(
            "k3",
            "model",
            URL,
            target.worker_revision()
        ));
        let snapshot = controller.distribution_headroom_snapshot("k3", "model");
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].issuable_slots(), 0);
        drop(lease);
        assert!(controller
            .active_distribution_targets("k3", "model")
            .is_empty());
        assert!(!controller.distribution_target_is_active(
            "k3",
            "model",
            URL,
            target.worker_revision()
        ));
        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        assert!(acquire_distribution_headroom(&controller, &target).is_some());
    }

    #[test]
    fn ordinary_predispatch_claim_hands_off_to_worker_load_without_slot_gap() {
        const URL: &str = "grpc://predispatch:30000";
        let registry = Arc::new(WorkerRegistry::new());
        let worker = register_headroom_worker(&registry, URL);
        let controller = AdaptiveAdmissionController::new(distribution_config(1), registry);
        update_worker_loads(&controller, [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 1)]))]);

        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        let selection = controller.lock_distribution_selection();
        let ordinary = controller
            .try_reserve_ordinary_distribution_target(&selection, "k3", "model", &worker)
            .unwrap();
        drop(selection);
        assert_eq!(
            controller
                .distribution_headroom_snapshot("k3", "model")
                .pop()
                .unwrap()
                .issuable_slots(),
            0
        );
        let seed_selection = controller.lock_distribution_selection();
        assert!(controller
            .try_acquire_distribution_headroom(&seed_selection, &target)
            .is_none());
        drop(seed_selection);

        let load_guard = WorkerLoadGuard::new(Arc::clone(&worker), None);
        drop(ordinary);
        assert_eq!(
            controller
                .distribution_headroom_snapshot("k3", "model")
                .pop()
                .unwrap()
                .issuable_slots(),
            0,
            "live worker load must cover the exact handoff after claim release"
        );
        drop(load_guard);
        assert!(controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .is_some_and(|candidate| candidate.issuable_slots() == 1));
    }

    #[test]
    fn distribution_headroom_runtime_rejects_seed_caps_above_one() {
        const URL: &str = "grpc://unsupported-cap:30000";
        let registry = Arc::new(WorkerRegistry::new());
        register_headroom_worker(&registry, URL);
        let controller = AdaptiveAdmissionController::new(distribution_config(2), registry);
        update_worker_loads(&controller, [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 8)]))]);

        assert!(!controller.distribution_headroom_enabled("k3"));
        assert!(controller
            .distribution_headroom_snapshot("k3", "model")
            .is_empty());
    }

    #[test]
    fn distribution_headroom_revision_health_and_model_changes_fail_closed() {
        const URL: &str = "grpc://revision:30000";
        let registry = Arc::new(WorkerRegistry::new());
        let original = register_headroom_worker(&registry, URL);
        let worker_id = registry.get_id_by_url(URL).unwrap();
        let controller =
            AdaptiveAdmissionController::new(distribution_config(1), Arc::clone(&registry));
        let load = worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]);
        update_worker_loads(&controller, [(URL, load.clone())]);

        assert!(controller
            .distribution_headroom_snapshot("k3", "model-alias")
            .is_empty());
        let stale_target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        assert!(stale_target.matches_worker(&original));
        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(URL)
                .model(ModelCard::new("model").with_alias("model-alias"))
                .label(ADMISSION_PARTITION_LABEL, "k3")
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        assert!(!stale_target.matches_worker(&replacement));
        assert!(registry.replace(&worker_id, replacement));
        assert!(acquire_distribution_headroom(&controller, &stale_target).is_none());

        update_worker_loads(&controller, [(URL, load.clone())]);
        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        let lease = acquire_distribution_headroom(&controller, &target).unwrap();
        assert!(lease.verify());
        let second_replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(URL)
                .model(ModelCard::new("model").with_alias("model-alias"))
                .label(ADMISSION_PARTITION_LABEL, "k3")
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        assert!(registry.replace(&worker_id, second_replacement));
        assert!(
            !lease.verify(),
            "same-ID replacement must invalidate exact worker-instance identity"
        );
        drop(lease);

        update_worker_loads(&controller, [(URL, load.clone())]);
        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        let lease = acquire_distribution_headroom(&controller, &target).unwrap();
        assert!(lease.verify());
        update_worker_loads(&controller, [(URL, load)]);
        assert!(
            !lease.verify(),
            "telemetry revision changes must invalidate"
        );
        drop(lease);

        update_worker_loads(
            &controller,
            [(URL, worker_load(&[(0, 0, 0, 0.1, 0.1, 16)]))],
        );
        let target = controller
            .distribution_headroom_snapshot("k3", "model")
            .pop()
            .unwrap();
        registry
            .get_by_url(URL)
            .unwrap()
            .set_status(WorkerStatus::NotReady);
        assert!(acquire_distribution_headroom(&controller, &target).is_none());
        assert!(!registry.get_by_url(URL).unwrap().is_available());
    }

    #[test]
    fn feedback_knee_moves_down_on_same_throughput_plateau() {
        let start = Instant::now();
        let mut estimate = FeedbackEstimate {
            peak_tokens_per_second_per_replica: 100.0,
            running_requests_per_replica_at_peak: 20.0,
            pressure_observed: false,
            last_update: start,
        };
        estimate.observe(99.0, 12.0, start + Duration::from_secs(1), 60.0, 0.02, true);
        assert_eq!(estimate.running_requests_per_replica_at_peak, 12.0);
        assert!(estimate.pressure_observed);
    }
}
