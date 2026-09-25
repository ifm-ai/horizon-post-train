use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};

use kv_index::compute_request_content_hashes;
use metrics::{counter, histogram};
use openai_protocol::worker::WorkerLoadResponse;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{
    config::LeastLoadCacheMode,
    worker::{KvEventMonitor, Worker},
};

/// Default KV-pressure weight `λ_t` (seconds): the time-cost of KV contention,
/// chosen commensurate with the expected-queue-wait term so the two add cleanly.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;

/// Default mean prefill length (tokens), used to estimate in-flight token-work
/// for a dispatched request whose token count is unknown at routing time.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;

/// Default fallback throughput (tokens/s) for the `/throughput` term when a
/// backend reports KV usage but no live `gen_throughput`. On a homogeneous
/// fleet its absolute value mainly sets the work-vs-barrier balance, so it
/// co-tunes with `kv_pressure_weight`.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;
pub const DEFAULT_CACHE_PREFILL_THROUGHPUT: f64 = 8000.0;
pub const DEFAULT_MEAN_REMAINING_DECODE_TOKENS: u32 = 2048;

/// The production TokenSpeed KV stream has no heartbeat. Do not use a
/// positive ownership claim after a quiet stream has exceeded this bound.
const CACHE_EVENT_MAX_AGE: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct PreparedCacheCredit {
    monitor: Arc<KvEventMonitor>,
    /// One entry per input worker. `Some` means the worker had a fresh,
    /// certified stream at query time, even when its exact overlap is zero.
    claims: Vec<Option<(u64, u64)>>,
}

#[derive(Debug, Default)]
struct InflightState {
    /// Legacy token credits for the worker actually dispatched.
    legacy_tokens: HashMap<String, u64>,
    /// Counterfactual (Shadow) or actual (Enforce) bounded-policy credits.
    /// These use the same token-work units as the legacy score.
    cache_tokens: HashMap<String, u64>,
}

#[derive(Clone, Copy, Debug)]
enum CacheCreditSkipReason {
    InsufficientWorkers,
    MissingPromptTokens,
    UnsupportedTopology,
    MonitorUnavailable,
    BlockSizeUnavailable,
    IndexerUnavailable,
    NoContentBlocks,
    NoCertifiedStreams,
    StaleKv,
    MissingLoad,
    NoPrefix,
}

impl CacheCreditSkipReason {
    fn as_label(self) -> &'static str {
        match self {
            Self::InsufficientWorkers => "insufficient_workers",
            Self::MissingPromptTokens => "missing_prompt_tokens",
            Self::UnsupportedTopology => "unsupported_topology",
            Self::MonitorUnavailable => "monitor_unavailable",
            Self::BlockSizeUnavailable => "block_size_unavailable",
            Self::IndexerUnavailable => "indexer_unavailable",
            Self::NoContentBlocks => "no_content_blocks",
            Self::NoCertifiedStreams => "no_certified_streams",
            Self::StaleKv => "stale_kv",
            Self::MissingLoad => "missing_load",
            Self::NoPrefix => "no_prefix",
        }
    }
}

/// Least-(token-)work routing — route to the worker with the lowest estimated
/// time-to-drain plus a convex KV-pressure barrier (argmin, lower is better):
///
/// ```text
///   score_i = (queued_tokens_i + inflight_tokens_i) / throughput_i
///             + kv_pressure_weight · k_i / (1 − k_i)
/// ```
///
/// - `queued_tokens` — the backend's waiting-queue token-work
///   (`num_waiting_uncached_tokens`). Token-work, not request count, is what
///   sets the wait under size-skewed traffic: a long prompt is far more work
///   than a short one, regardless of how many requests are queued.
/// - `inflight_tokens` — token-work this router has dispatched to the worker
///   since its last load poll. Polls are stale between intervals; without this
///   correction, plain argmin sends a whole interval's arrivals to one worker
///   (incast). Crediting each dispatch water-fills load across workers instead.
/// - `/ throughput` — normalizes work to *time*, comparing heterogeneous
///   workers by drain time rather than raw token count.
/// - `k / (1 − k)` — the M/M/1 expected-occupancy barrier on KV utilization
///   `k`; convex and divergent at the KV cliff, so routing avoids the
///   preemption/recompute that begins as KV fills.
///
/// Both terms are in seconds, so they add directly. Missing signals degrade
/// gracefully and stay in time units:
/// - no queued-token report (backend doesn't expose waiting-queue tokens):
///   `queued_tokens = 0`, leaving in-flight-corrected drain time plus the barrier;
/// - zero/absent throughput (backend reports no generation rate): falls back to
///   the configured `default_throughput`, so the work term stays in seconds and
///   the KV barrier stays relevant;
/// - a worker with no fresh snapshot while peers report: its live in-flight is
///   converted to a drain-time estimate (`load · p̄ / fleet_nominal_throughput`)
///   so it is comparable to reporting workers, not scored on a raw count;
/// - the whole fleet dark (true cold start, or a backend that never reports
///   loads): join-shortest-queue on the live in-flight count.
///
/// In-flight token-work is exact on the gRPC routing path (the request's token
/// count is known at selection); the HTTP path has no token count and falls
/// back to `p̄ · count`, which is weaker on size-skewed traffic. This policy is
/// therefore intended for gRPC workers.
///
/// # Tuning knobs
///
/// All are fields of `PolicyConfig::LeastLoad` with the defaults below:
/// - `kv_pressure_weight` (λ_t, default `0.15` s) — weight of the KV-pressure
///   barrier. Raise it to steer harder away from near-full KV; lower it to
///   weight raw drain time more.
/// - `default_throughput` (default `2000` tok/s) — drain rate used when a
///   backend reports no live `gen_throughput`. Set it to the fleet's measured
///   per-replica generation rate; it co-tunes with `kv_pressure_weight`.
/// - `mean_prefill_tokens` (p̄, default `1024`) — per-request token estimate for
///   the in-flight term when the request's token count is unknown at routing
///   (the HTTP path; ignored when tokens are known, i.e. gRPC).
/// - `load_check_interval_secs` (default `10`) — worker-load poll period; the
///   in-flight correction absorbs staleness between polls.
#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// Cached load reports from the worker monitor (keyed by worker URL).
    cached_loads: RwLock<HashMap<String, WorkerLoadResponse>>,
    /// In-flight token-work dispatched per worker since its last load poll
    /// (keyed by worker URL); reset when a fresh report arrives.
    /// Both since-poll ledgers share one lock so routing and poll reset cannot
    /// expose a half-updated legacy/cache-aware state.
    inflight: RwLock<InflightState>,
    /// KV-pressure weight `λ_t` (seconds).
    kv_pressure_weight: f64,
    /// Mean prefill length (tokens) for estimating in-flight token-work when a
    /// request's token count is unknown at routing time.
    mean_prefill_tokens: u32,
    /// Fallback throughput (tokens/s) for the `/throughput` term when a backend
    /// reports no live `gen_throughput`.
    default_throughput: f64,
    cache_mode: LeastLoadCacheMode,
    cache_prefill_throughput: f64,
    kv_event_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_KV_PRESSURE_WEIGHT,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        )
    }

    pub fn with_kv_pressure_weight(kv_pressure_weight: f64) -> Self {
        Self::with_params(
            kv_pressure_weight,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        )
    }

    pub fn with_params(
        kv_pressure_weight: f64,
        mean_prefill_tokens: u32,
        default_throughput: f64,
    ) -> Self {
        Self::with_cache_params(
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
            LeastLoadCacheMode::Off,
            DEFAULT_CACHE_PREFILL_THROUGHPUT,
            DEFAULT_MEAN_REMAINING_DECODE_TOKENS,
        )
    }

    pub fn with_cache_params(
        kv_pressure_weight: f64,
        mean_prefill_tokens: u32,
        default_throughput: f64,
        cache_mode: LeastLoadCacheMode,
        cache_prefill_throughput: f64,
        _mean_remaining_decode_tokens: u32,
    ) -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
            inflight: RwLock::new(InflightState::default()),
            kv_pressure_weight: if kv_pressure_weight.is_finite() && kv_pressure_weight >= 0.0 {
                kv_pressure_weight
            } else {
                DEFAULT_KV_PRESSURE_WEIGHT
            },
            mean_prefill_tokens: mean_prefill_tokens.max(1),
            default_throughput: if default_throughput.is_finite() && default_throughput > 0.0 {
                default_throughput
            } else {
                DEFAULT_THROUGHPUT
            },
            cache_mode,
            cache_prefill_throughput: if cache_prefill_throughput.is_finite()
                && cache_prefill_throughput > 0.0
            {
                cache_prefill_throughput
            } else {
                DEFAULT_CACHE_PREFILL_THROUGHPUT
            },
            kv_event_monitor: RwLock::new(None),
        }
    }

    /// Expected-wait score for a worker (lower is better).
    ///
    /// `inflight` maps worker URL -> token-work dispatched since its last poll.
    /// `nominal_throughput` (a peer-derived mean) estimates drain rate for a
    /// worker missing a fresh snapshot; `fleet_has_loads` is false only when no
    /// worker reports at all, in which case we fall back to join-shortest-queue
    /// on the live in-flight count (which, unlike the since-poll estimate,
    /// reflects completions and so suits backends that never report loads).
    fn score(
        &self,
        worker: &Arc<dyn Worker>,
        loads: Option<&HashMap<String, WorkerLoadResponse>>,
        inflight: &HashMap<String, u64>,
        nominal_throughput: f64,
        fleet_has_loads: bool,
    ) -> f64 {
        let url = worker.url();
        match loads.and_then(|m| m.get(url)) {
            Some(load) => {
                let inflight_tokens = inflight.get(url).copied().unwrap_or(0) as f64;
                let queued_tokens = load.total_waiting_uncached_tokens() as f64;
                let live_throughput = load.total_gen_throughput();
                let throughput = if live_throughput > 0.0 {
                    live_throughput
                } else {
                    self.default_throughput
                };
                let k = load.effective_token_usage().clamp(0.0, 0.999);
                (queued_tokens + inflight_tokens) / throughput
                    + self.kv_pressure_weight * k / (1.0 - k)
            }
            // No fresh snapshot, but peers report: estimate this worker's drain
            // time from its live in-flight (count × mean prefill) at the fleet's
            // nominal throughput, keeping the same units as reporting workers.
            None if fleet_has_loads => {
                worker.load() as f64 * self.mean_prefill_tokens as f64 / nominal_throughput
            }
            // Whole fleet dark (cold start, or a backend that never reports
            // loads): join-shortest-queue on live in-flight.
            None => worker.load() as f64,
        }
    }

    /// Token-work the request being routed adds to the chosen worker's
    /// in-flight estimate: its token count if known, else the mean prefill.
    fn request_tokens(&self, info: &SelectWorkerInfo) -> u64 {
        info.tokens
            .map(|t| t.len() as u64)
            .unwrap_or(self.mean_prefill_tokens as u64)
    }

    /// The unchanged legacy selection and credit operation, with the caller
    /// holding the combined since-poll state lock.
    fn select_legacy_locked(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        info: &SelectWorkerInfo<'_>,
        loads: Option<&HashMap<String, WorkerLoadResponse>>,
        inflight: &mut InflightState,
    ) -> Option<usize> {
        if healthy.is_empty() {
            return None;
        }
        // Preserve the production fast path exactly: one worker is returned
        // without a legacy since-poll credit or processed counter increment.
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let (tp_sum, tp_count) = healthy
            .iter()
            .filter_map(|&idx| loads.and_then(|loads| loads.get(workers[idx].url())))
            .map(WorkerLoadResponse::total_gen_throughput)
            .filter(|throughput| *throughput > 0.0)
            .fold((0.0, 0u32), |(sum, count), throughput| {
                (sum + throughput, count + 1)
            });
        let nominal_throughput = if tp_count > 0 {
            tp_sum / tp_count as f64
        } else {
            self.default_throughput
        };
        let fleet_has_loads = loads.is_some_and(|loads| {
            healthy
                .iter()
                .any(|&idx| loads.contains_key(workers[idx].url()))
        });

        let mut best = healthy[0];
        let mut best_score = self.score(
            &workers[best],
            loads,
            &inflight.legacy_tokens,
            nominal_throughput,
            fleet_has_loads,
        );
        for &idx in &healthy[1..] {
            let score = self.score(
                &workers[idx],
                loads,
                &inflight.legacy_tokens,
                nominal_throughput,
                fleet_has_loads,
            );
            if score < best_score {
                best = idx;
                best_score = score;
            }
        }

        let request_tokens = self.request_tokens(info);
        *inflight
            .legacy_tokens
            .entry(workers[best].url().to_string())
            .or_insert(0) += request_tokens;
        debug!(
            "least_load selected {} (score {:.4}, in_flight {})",
            workers[best].url(),
            best_score,
            workers[best].load()
        );
        workers[best].increment_processed();
        Some(best)
    }

    pub fn cache_credit_enabled(&self) -> bool {
        self.cache_mode != LeastLoadCacheMode::Off
    }

    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self
            .kv_event_monitor
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = monitor;
    }

    fn mode_label(&self) -> &'static str {
        match self.cache_mode {
            LeastLoadCacheMode::Off => "off",
            LeastLoadCacheMode::Shadow => "shadow",
            LeastLoadCacheMode::Enforce => "enforce",
        }
    }

    fn record_cache_decision(
        &self,
        model_id: &str,
        result: &'static str,
        changed: bool,
        cache_savings_seconds: Option<f64>,
    ) {
        counter!(
            "smg_least_load_cache_credit_decisions_total",
            "mode" => self.mode_label(),
            "model" => model_id.to_string(),
            "result" => result,
            "changed" => if changed { "true" } else { "false" },
        )
        .increment(1);
        if let Some(seconds) = cache_savings_seconds {
            histogram!(
                "smg_least_load_cache_savings_seconds",
                "mode" => self.mode_label(),
                "model" => model_id.to_string(),
            )
            .record(seconds);
        }
    }

    fn record_cache_skip(&self, model_id: &str, reason: CacheCreditSkipReason) {
        counter!(
            "smg_least_load_cache_credit_skips_total",
            "mode" => self.mode_label(),
            "model" => model_id.to_string(),
            "reason" => reason.as_label(),
        )
        .increment(1);
    }

    fn prepare_cache_credit(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        info: &SelectWorkerInfo<'_>,
        model_id: &str,
    ) -> Result<PreparedCacheCredit, CacheCreditSkipReason> {
        let tokens = info
            .tokens
            .filter(|tokens| !tokens.is_empty())
            .ok_or(CacheCreditSkipReason::MissingPromptTokens)?;

        // KV events do not carry a model ID. Cache credit is safe only when
        // every candidate advertises exactly the requested single model.
        if healthy.iter().any(|&idx| {
            let models = workers[idx].models();
            models.len() != 1 || !models[0].matches(model_id)
        }) {
            return Err(CacheCreditSkipReason::UnsupportedTopology);
        }

        let monitor = self
            .kv_event_monitor
            .read()
            .map_err(|_| CacheCreditSkipReason::MonitorUnavailable)?
            .as_ref()
            .map(Arc::clone)
            .ok_or(CacheCreditSkipReason::MonitorUnavailable)?;
        let block_size = monitor
            .block_size(model_id)
            .filter(|size| *size > 0)
            .ok_or(CacheCreditSkipReason::BlockSizeUnavailable)?;
        let indexer = monitor
            .get_indexer(model_id)
            .ok_or(CacheCreditSkipReason::IndexerUnavailable)?;
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        if content_hashes.is_empty() {
            return Err(CacheCreditSkipReason::NoContentBlocks);
        }

        // Capture generations before the exact index query, then validate
        // again after it. Unknown/disconnected workers remain safe zero-cache
        // candidates, while at least one certified stream keeps routing active.
        let mut generations = vec![None; workers.len()];
        for &idx in healthy {
            generations[idx] =
                monitor.certified_generation(workers[idx].url(), CACHE_EVENT_MAX_AGE);
        }
        let scores = indexer.find_certified_matches(&content_hashes);
        let mut claims = vec![None; workers.len()];
        let mut certified = 0usize;
        for &idx in healthy {
            let Some(generation) = generations[idx] else {
                continue;
            };
            let url = workers[idx].url();
            let Some(worker_id) = indexer.worker_id(url) else {
                continue;
            };
            if !monitor.certified_generation_unchanged(url, generation, CACHE_EVENT_MAX_AGE) {
                continue;
            }
            let cached_blocks = scores.scores.get(&worker_id).copied().unwrap_or(0) as u64;
            let cached_tokens = cached_blocks
                .saturating_mul(block_size as u64)
                .min(tokens.len() as u64);
            claims[idx] = Some((generation, cached_tokens));
            certified += 1;
        }
        if certified == 0 {
            return Err(CacheCreditSkipReason::NoCertifiedStreams);
        }
        Ok(PreparedCacheCredit { monitor, claims })
    }

    /// Cache credit requires a structurally valid direct load sample. A worker
    /// without one still participates in ordinary least-load selection, but it
    /// cannot receive a cache discount for this decision.
    fn cache_load_usable(load: &WorkerLoadResponse) -> bool {
        let Ok(rank_count) = usize::try_from(load.dp_rank_count) else {
            return false;
        };
        if rank_count == 0 || rank_count != load.loads.len() {
            return false;
        }
        for rank in &load.loads {
            if rank.num_running_reqs < 0
                || rank.num_waiting_reqs < 0
                || rank.num_waiting_uncached_tokens < 0
            {
                return false;
            }
        }
        let live_throughput = load.total_gen_throughput();
        live_throughput.is_finite() && live_throughput >= 0.0
    }

    fn fallback_legacy_with_cache_credit(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        let loads_guard = self.cached_loads.read().ok();
        let loads = loads_guard.as_deref();
        let mut inflight = self
            .inflight
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let selected = self.select_legacy_locked(workers, &healthy, info, loads, &mut inflight)?;
        let request_tokens = self.request_tokens(info);
        *inflight
            .cache_tokens
            .entry(workers[selected].url().to_string())
            .or_insert(0) += request_tokens;
        Some(selected)
    }

    /// Narrow rollout path: direct Regular single-model streaming gRPC Chat.
    /// Every unavailable signal delegates to the unchanged legacy method.
    pub fn select_worker_cache_credit_chat(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        model_id: &str,
    ) -> Option<usize> {
        if self.cache_mode == LeastLoadCacheMode::Off {
            return self.select_worker(workers, info);
        }

        let healthy = get_healthy_worker_indices(workers);
        if healthy.len() <= 1 {
            self.record_cache_decision(model_id, "fallback", false, None);
            self.record_cache_skip(model_id, CacheCreditSkipReason::InsufficientWorkers);
            return self.fallback_legacy_with_cache_credit(workers, info);
        }
        let prepared = match self.prepare_cache_credit(workers, &healthy, info, model_id) {
            Ok(prepared) => prepared,
            Err(reason) => {
                self.record_cache_decision(model_id, "fallback", false, None);
                self.record_cache_skip(model_id, reason);
                return self.fallback_legacy_with_cache_credit(workers, info);
            }
        };

        let loads_guard = self.cached_loads.read().ok();
        let Some(loads) = loads_guard.as_deref() else {
            self.record_cache_decision(model_id, "fallback", false, None);
            self.record_cache_skip(model_id, CacheCreditSkipReason::MissingLoad);
            return self.fallback_legacy_with_cache_credit(workers, info);
        };
        if !healthy
            .iter()
            .any(|&idx| loads.contains_key(workers[idx].url()))
        {
            drop(loads_guard);
            self.record_cache_decision(model_id, "fallback", false, None);
            self.record_cache_skip(model_id, CacheCreditSkipReason::MissingLoad);
            return self.fallback_legacy_with_cache_credit(workers, info);
        }

        let (tp_sum, tp_count) = healthy
            .iter()
            .filter_map(|&idx| loads.get(workers[idx].url()))
            .map(WorkerLoadResponse::total_gen_throughput)
            .filter(|throughput| *throughput > 0.0)
            .fold((0.0, 0u32), |(sum, count), throughput| {
                (sum + throughput, count + 1)
            });
        let nominal_throughput = if tp_count > 0 {
            tp_sum / tp_count as f64
        } else {
            self.default_throughput
        };

        let mut inflight = self
            .inflight
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Revalidate after taking the selection lock. A disconnect, gap, or
        // restart between hashing and this point converts that worker to a
        // safe zero-cache candidate. If every stream changed, fall back.
        let mut cached_tokens = vec![0u64; workers.len()];
        let mut certified = 0usize;
        let mut stale_kv = false;
        for &idx in &healthy {
            let Some((generation, tokens)) = prepared.claims[idx] else {
                continue;
            };
            if prepared.monitor.certified_generation_unchanged(
                workers[idx].url(),
                generation,
                CACHE_EVENT_MAX_AGE,
            ) {
                cached_tokens[idx] = tokens;
                certified += 1;
            } else {
                stale_kv = true;
            }
        }
        if certified == 0 {
            drop(inflight);
            drop(loads_guard);
            self.record_cache_decision(model_id, "fallback", false, None);
            self.record_cache_skip(model_id, CacheCreditSkipReason::StaleKv);
            return self.fallback_legacy_with_cache_credit(workers, info);
        }

        let legacy_score = |idx: usize| {
            self.score(
                &workers[idx],
                Some(loads),
                &inflight.legacy_tokens,
                nominal_throughput,
                true,
            )
        };
        let cache_score = |idx: usize| {
            let base = self.score(
                &workers[idx],
                Some(loads),
                &inflight.cache_tokens,
                nominal_throughput,
                true,
            );
            let discount = loads
                .get(workers[idx].url())
                .filter(|load| Self::cache_load_usable(load))
                .map_or(0.0, |_| {
                    cached_tokens[idx] as f64 / self.cache_prefill_throughput
                });
            base - discount
        };

        let mut legacy_best = healthy[0];
        let mut legacy_best_score = legacy_score(legacy_best);
        let mut cache_best = healthy[0];
        let mut cache_best_score = cache_score(cache_best);
        for &idx in &healthy[1..] {
            let legacy = legacy_score(idx);
            if legacy < legacy_best_score {
                legacy_best = idx;
                legacy_best_score = legacy;
            }
            let cache = cache_score(idx);
            if cache < cache_best_score {
                cache_best = idx;
                cache_best_score = cache;
            }
        }

        let selected = if self.cache_mode == LeastLoadCacheMode::Enforce {
            cache_best
        } else {
            legacy_best
        };
        // Preserve the legacy since-poll ledger for the worker actually
        // dispatched in both modes. If KV certification disappears before the
        // next poll, fallback sees every Enforce dispatch already in flight.
        let req_tokens = self.request_tokens(info);
        *inflight
            .legacy_tokens
            .entry(workers[selected].url().to_string())
            .or_insert(0) += req_tokens;
        *inflight
            .cache_tokens
            .entry(workers[cache_best].url().to_string())
            .or_insert(0) += req_tokens;
        let skipped_missing_load = healthy.iter().any(|&idx| {
            prepared.claims[idx].is_some()
                && match loads.get(workers[idx].url()) {
                    Some(load) => !Self::cache_load_usable(load),
                    None => true,
                }
        });
        let no_cached_prefix = healthy.iter().all(|&idx| cached_tokens[idx] == 0);
        drop(inflight);
        drop(loads_guard);

        if skipped_missing_load {
            self.record_cache_skip(model_id, CacheCreditSkipReason::MissingLoad);
        }
        if stale_kv {
            self.record_cache_skip(model_id, CacheCreditSkipReason::StaleKv);
        }
        if no_cached_prefix {
            self.record_cache_skip(model_id, CacheCreditSkipReason::NoPrefix);
        }

        let result = if certified == healthy.len() {
            "complete"
        } else {
            "partial"
        };
        self.record_cache_decision(
            model_id,
            result,
            cache_best != legacy_best,
            Some(cached_tokens[cache_best] as f64 / self.cache_prefill_throughput),
        );
        debug!(
            "least_load cache credit selected {} (legacy {}, bounded_score {:.4})",
            workers[selected].url(),
            workers[legacy_best].url(),
            cache_best_score
        );
        workers[selected].increment_processed();
        Some(selected)
    }

    fn apply_load_update(
        &self,
        loads: &HashMap<String, WorkerLoadResponse>,
        worker_urls: Option<&[String]>,
    ) {
        if let Ok(mut cached) = self.cached_loads.write() {
            if let Ok(mut inflight) = self.inflight.write() {
                cached.extend(loads.iter().map(|(url, load)| (url.clone(), load.clone())));
                if let Some(worker_urls) = worker_urls {
                    cached.retain(|url, _| !worker_urls.contains(url) || loads.contains_key(url));
                }
                // A fresh snapshot already reflects work up to the poll. Only
                // reported URLs retire credits; a missing URL is evicted from
                // projected telemetry but keeps its conservative credits.
                for url in loads.keys() {
                    inflight.legacy_tokens.insert(url.clone(), 0);
                    inflight.cache_tokens.insert(url.clone(), 0);
                }
            }
        }
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let loads_guard = self.cached_loads.read().ok();
        let loads = loads_guard.as_deref();
        let mut inflight = self
            .inflight
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.select_legacy_locked(workers, &healthy, info, loads, &mut inflight)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        self.apply_load_update(loads, None);
    }

    fn update_loads_for_workers(
        &self,
        loads: &HashMap<String, WorkerLoadResponse>,
        worker_urls: &[String],
    ) {
        self.apply_load_update(loads, Some(worker_urls));
    }

    fn needs_load_updates(&self) -> bool {
        true
    }

    fn needs_kv_events(&self) -> bool {
        self.cache_credit_enabled()
    }

    fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        LeastLoadPolicy::set_kv_event_monitor(self, monitor);
    }

    fn remove_worker(&self, url: &str) {
        if let Ok(mut cached) = self.cached_loads.write() {
            cached.remove(url);
            if let Ok(mut inflight) = self.inflight.write() {
                inflight.legacy_tokens.remove(url);
                inflight.cache_tokens.remove(url);
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LeastLoadPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{mpsc, Arc as StdArc, Barrier},
    };

    use kv_index::{
        compute_content_hash, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap,
    };
    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, SchedulerLoadSnapshot},
    };

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    /// One DP rank with the given queued tokens, KV utilization, and throughput.
    fn make_load(
        num_waiting_uncached_tokens: i32,
        token_usage: f64,
        gen_throughput: f64,
    ) -> WorkerLoadResponse {
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 0,
                token_usage,
                gen_throughput,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                max_running_requests: 0,
                ..Default::default()
            }],
        }
    }

    fn mk(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn mk_model(url: &str, model_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new(model_id))
                .health_config(no_health_check())
                .build(),
        )
    }

    fn make_m2_load(running: i32, waiting: i32, gen_throughput: f64) -> WorkerLoadResponse {
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: running,
                num_waiting_reqs: waiting,
                num_waiting_uncached_tokens: 0,
                num_total_reqs: running.saturating_add(waiting),
                gen_throughput,
                ..Default::default()
            }],
        }
    }

    fn cache_policy(
        mode: LeastLoadCacheMode,
        default_throughput: f64,
        cache_prefill_throughput: f64,
        mean_remaining_decode_tokens: u32,
    ) -> LeastLoadPolicy {
        LeastLoadPolicy::with_cache_params(
            0.0,
            1024,
            default_throughput,
            mode,
            cache_prefill_throughput,
            mean_remaining_decode_tokens,
        )
    }

    fn certified_monitor(
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        block_size: usize,
        ready: &[usize],
        cached_owner: Option<(usize, &[u32])>,
    ) -> Arc<KvEventMonitor> {
        let monitor = Arc::new(KvEventMonitor::new(Some(block_size)));
        monitor.set_block_size(model_id, block_size);
        let indexer = Arc::new(PositionalIndexer::new(block_size));
        let mut worker_ids = Vec::with_capacity(workers.len());
        for worker in workers {
            worker_ids.push(indexer.intern_worker(worker.url()).unwrap());
        }
        if let Some((owner, tokens)) = cached_owner {
            let blocks: Vec<_> = tokens
                .chunks(block_size)
                .enumerate()
                .map(|(position, block)| StoredBlock {
                    seq_hash: SequenceHash(position as u64 + 1),
                    content_hash: compute_content_hash(block),
                })
                .collect();
            indexer
                .apply_stored(
                    worker_ids[owner],
                    &blocks,
                    None,
                    &mut WorkerBlockMap::default(),
                )
                .unwrap();
        }
        monitor
            .indexers
            .insert(model_id.to_string(), Arc::clone(&indexer));
        for &idx in ready {
            monitor.mark_event_stream_ready_for_test(workers[idx].url());
        }
        monitor
    }

    #[test]
    fn cache_credit_break_even_is_one_cached_prompt_second() {
        fn choose(owner_waiting_tokens: i32) -> usize {
            let model = "kimi-k3";
            let tokens = vec![7u32; 1024];
            let workers = vec![
                mk_model("http://owner:8000", model),
                mk_model("http://cold:8000", model),
            ];
            let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 1);
            policy.set_kv_event_monitor(Some(certified_monitor(
                model,
                &workers,
                16,
                &[0, 1],
                Some((0, &tokens)),
            )));
            let mut loads = HashMap::new();
            loads.insert(
                workers[0].url().to_string(),
                make_load(owner_waiting_tokens, 0.0, 1000.0),
            );
            loads.insert(workers[1].url().to_string(), make_load(0, 0.0, 1000.0));
            policy.update_loads(&loads);
            policy
                .select_worker_cache_credit_chat(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&tokens),
                        ..Default::default()
                    },
                    model,
                )
                .unwrap()
        }

        // 1024 cached tokens at 1000 prefill tok/s save 1.024 seconds.
        // The cache owner wins only while that saving covers its additional
        // ordinary least-load queue cost.
        assert_eq!(choose(1023), 0, "1.023s penalty keeps the cache owner");
        assert_eq!(choose(1025), 1, "1.025s penalty spills to the cold peer");
    }

    #[test]
    fn zero_ready_cache_streams_fall_back_to_exact_legacy_sequence() {
        let model = "kimi-k3";
        let tokens = vec![1u32; 128];
        let legacy_workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let projected_workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let legacy = cache_policy(LeastLoadCacheMode::Off, 1000.0, 1000.0, 2048);
        let projected = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 2048);
        projected.set_kv_event_monitor(Some(certified_monitor(
            model,
            &projected_workers,
            16,
            &[],
            Some((0, &tokens)),
        )));
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_m2_load(0, 0, 0.0));
        loads.insert("http://b:8000".to_string(), make_m2_load(0, 0, 0.0));
        legacy.update_loads(&loads);
        projected.update_loads(&loads);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let legacy_sequence: Vec<_> = (0..20)
            .map(|_| legacy.select_worker(&legacy_workers, &info).unwrap())
            .collect();
        let projected_sequence: Vec<_> = (0..20)
            .map(|_| {
                projected
                    .select_worker_cache_credit_chat(&projected_workers, &info, model)
                    .unwrap()
            })
            .collect();
        assert_eq!(projected_sequence, legacy_sequence);
        assert_eq!(
            projected.inflight.read().unwrap().legacy_tokens,
            legacy.inflight.read().unwrap().legacy_tokens
        );
    }

    #[test]
    fn stale_cache_streams_fall_back_to_exact_legacy_sequence() {
        let model = "kimi-k3";
        let tokens = vec![2u32; 128];
        let legacy_workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let projected_workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let legacy = cache_policy(LeastLoadCacheMode::Off, 1000.0, 1000.0, 2048);
        let projected = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 2048);
        let monitor = certified_monitor(model, &projected_workers, 16, &[0, 1], Some((0, &tokens)));
        for worker in &projected_workers {
            monitor.age_event_stream_for_test(worker.url(), Duration::from_secs(31));
        }
        projected.set_kv_event_monitor(Some(monitor));
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_m2_load(0, 0, 0.0));
        loads.insert("http://b:8000".to_string(), make_m2_load(0, 0, 0.0));
        legacy.update_loads(&loads);
        projected.update_loads(&loads);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let legacy_sequence: Vec<_> = (0..20)
            .map(|_| legacy.select_worker(&legacy_workers, &info).unwrap())
            .collect();
        let projected_sequence: Vec<_> = (0..20)
            .map(|_| {
                projected
                    .select_worker_cache_credit_chat(&projected_workers, &info, model)
                    .unwrap()
            })
            .collect();
        assert_eq!(projected_sequence, legacy_sequence);
        assert_eq!(
            projected.inflight.read().unwrap().legacy_tokens,
            legacy.inflight.read().unwrap().legacy_tokens
        );
    }

    #[test]
    fn missing_current_group_load_evicts_stale_snapshot_and_falls_back() {
        let model = "kimi-k3";
        let tokens = vec![8u32; 128];
        let workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 100);
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            &workers,
            16,
            &[0, 1],
            Some((0, &tokens)),
        )));
        let mut full = HashMap::new();
        full.insert(workers[0].url().to_string(), make_m2_load(0, 0, 0.0));
        full.insert(workers[1].url().to_string(), make_m2_load(0, 0, 0.0));
        policy.update_loads(&full);

        let group_urls = workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect::<Vec<_>>();
        let only_a = HashMap::from([(workers[0].url().to_string(), make_m2_load(0, 0, 0.0))]);
        policy.update_loads_for_workers(&only_a, &group_urls);
        assert!(!policy
            .cached_loads
            .read()
            .unwrap()
            .contains_key(workers[1].url()));

        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let chosen = policy
            .select_worker_cache_credit_chat(&workers, &info, model)
            .unwrap();
        assert_eq!(
            chosen, 0,
            "the reporting cache owner remains eligible without a fleet-wide fallback"
        );
    }

    #[test]
    fn missing_load_excludes_only_that_worker_from_cache_credit() {
        let model = "kimi-k3";
        let tokens = vec![18u32; 8192];
        let owner = mk_model("http://owner:8000", model);
        let cold = mk_model("http://cold:8000", model);
        for _ in 0..5 {
            owner.increment_load();
        }
        let workers = vec![owner, cold];
        let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 1);
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            &workers,
            16,
            &[0, 1],
            Some((0, &tokens)),
        )));
        policy.update_loads(&HashMap::from([(
            workers[1].url().to_string(),
            make_m2_load(0, 0, 1000.0),
        )]));

        assert_eq!(
            policy.select_worker_cache_credit_chat(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
                model,
            ),
            Some(1),
            "an owner without direct load must not receive an 8.192 second cache discount"
        );
    }

    #[test]
    fn shadow_keeps_exact_legacy_sequence_with_certified_cache() {
        let model = "kimi-k3";
        let tokens = vec![3u32; 1024];
        let legacy_workers = vec![
            mk_model("http://owner:8000", model),
            mk_model("http://cold:8000", model),
        ];
        let shadow_workers = vec![
            mk_model("http://owner:8000", model),
            mk_model("http://cold:8000", model),
        ];
        let legacy = cache_policy(LeastLoadCacheMode::Off, 1000.0, 1000.0, 1);
        let shadow = cache_policy(LeastLoadCacheMode::Shadow, 1000.0, 1000.0, 1);
        shadow.set_kv_event_monitor(Some(certified_monitor(
            model,
            &shadow_workers,
            16,
            &[0, 1],
            Some((0, &tokens)),
        )));
        let mut loads = HashMap::new();
        loads.insert(
            "http://owner:8000".to_string(),
            make_load(1023, 0.0, 1000.0),
        );
        loads.insert("http://cold:8000".to_string(), make_load(0, 0.0, 1000.0));
        legacy.update_loads(&loads);
        shadow.update_loads(&loads);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let legacy_sequence: Vec<_> = (0..20)
            .map(|_| legacy.select_worker(&legacy_workers, &info).unwrap())
            .collect();
        let shadow_sequence: Vec<_> = (0..20)
            .map(|_| {
                shadow
                    .select_worker_cache_credit_chat(&shadow_workers, &info, model)
                    .unwrap()
            })
            .collect();
        assert_eq!(shadow_sequence, legacy_sequence);
        assert_eq!(
            shadow.inflight.read().unwrap().legacy_tokens,
            legacy.inflight.read().unwrap().legacy_tokens
        );
    }

    #[test]
    fn mean_remaining_decode_does_not_change_bounded_choice() {
        fn choose(mean_remaining_decode_tokens: u32) -> usize {
            let model = "kimi-k3";
            let tokens = vec![19u32; 1024];
            let workers = vec![
                mk_model("http://owner:8000", model),
                mk_model("http://cold:8000", model),
            ];
            let policy = cache_policy(
                LeastLoadCacheMode::Enforce,
                1000.0,
                1000.0,
                mean_remaining_decode_tokens,
            );
            policy.set_kv_event_monitor(Some(certified_monitor(
                model,
                &workers,
                16,
                &[0, 1],
                Some((0, &tokens)),
            )));
            policy.update_loads(&HashMap::from([
                (workers[0].url().to_string(), make_load(500, 0.0, 1000.0)),
                (workers[1].url().to_string(), make_load(0, 0.0, 1000.0)),
            ]));
            policy
                .select_worker_cache_credit_chat(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&tokens),
                        ..Default::default()
                    },
                    model,
                )
                .unwrap()
        }

        assert_eq!(choose(1), 0);
        assert_eq!(choose(65_535), 0);
    }

    #[test]
    fn enforce_dispatches_remain_visible_after_kv_fallback() {
        let model = "kimi-k3";
        let tokens = vec![5u32; 1024];
        let workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 100);
        let monitor = certified_monitor(model, &workers, 16, &[0, 1], Some((0, &tokens)));
        policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        let mut loads = HashMap::new();
        loads.insert(workers[0].url().to_string(), make_m2_load(0, 0, 1000.0));
        loads.insert(workers[1].url().to_string(), make_m2_load(0, 0, 1000.0));
        policy.update_loads(&loads);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let first = policy
            .select_worker_cache_credit_chat(&workers, &info, model)
            .unwrap();
        assert_eq!(
            policy.inflight.read().unwrap().legacy_tokens[workers[first].url()],
            tokens.len() as u64
        );
        for worker in &workers {
            monitor.mark_event_stream_not_ready_for_test(worker.url());
        }
        let second = policy
            .select_worker_cache_credit_chat(&workers, &info, model)
            .unwrap();
        assert_ne!(
            second, first,
            "legacy fallback must see the Enforce dispatch"
        );
    }

    #[test]
    fn one_disconnected_peer_keeps_safe_partial_cache_credit() {
        let model = "kimi-k3";
        let tokens = vec![9u32; 1024];
        let workers = vec![
            mk_model("http://owner:8000", model),
            mk_model("http://unknown:8000", model),
        ];
        let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 1000.0, 1);
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            &workers,
            16,
            &[0],
            Some((0, &tokens)),
        )));
        let mut loads = HashMap::new();
        loads.insert(workers[0].url().to_string(), make_m2_load(0, 0, 0.0));
        loads.insert(workers[1].url().to_string(), make_m2_load(0, 0, 0.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker_cache_credit_chat(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
                model,
            ),
            Some(0)
        );
    }

    #[test]
    fn cache_credit_requires_a_usable_direct_load() {
        assert!(LeastLoadPolicy::cache_load_usable(&make_m2_load(
            1, 0, 100.0
        )));
        assert!(LeastLoadPolicy::cache_load_usable(&make_m2_load(1, 0, 0.0)));
        assert!(!LeastLoadPolicy::cache_load_usable(&WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: 1,
            loads: vec![],
        }));
    }

    #[test]
    fn fully_cached_burst_still_waterfills_on_legacy_token_work() {
        let model = "kimi-k3";
        let tokens = vec![77u32; 8192];
        let workers = vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ];
        let policy = cache_policy(LeastLoadCacheMode::Enforce, 1000.0, 8000.0, 100);
        let monitor = certified_monitor(model, &workers, 16, &[0, 1], Some((0, &tokens)));
        let indexer = monitor.get_indexer(model).unwrap();
        let worker_b = indexer.worker_id(workers[1].url()).unwrap();
        let blocks: Vec<_> = tokens
            .chunks(16)
            .enumerate()
            .map(|(position, block)| StoredBlock {
                seq_hash: SequenceHash(position as u64 + 10_000),
                content_hash: compute_content_hash(block),
            })
            .collect();
        indexer
            .apply_stored(worker_b, &blocks, None, &mut WorkerBlockMap::default())
            .unwrap();
        policy.set_kv_event_monitor(Some(monitor));
        let mut loads = HashMap::new();
        loads.insert(workers[0].url().to_string(), make_m2_load(0, 0, 1000.0));
        loads.insert(workers[1].url().to_string(), make_m2_load(0, 0, 1000.0));
        policy.update_loads(&loads);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let selections: Vec<_> = (0..20)
            .map(|_| {
                policy
                    .select_worker_cache_credit_chat(&workers, &info, model)
                    .unwrap()
            })
            .collect();
        assert_eq!(selections.iter().filter(|&&idx| idx == 0).count(), 10);
        assert_eq!(selections.iter().filter(|&&idx| idx == 1).count(), 10);
        let inflight = policy.inflight.read().unwrap();
        assert_eq!(inflight.legacy_tokens.len(), 2);
        assert!(inflight.legacy_tokens.values().all(|tokens| *tokens > 0));
        assert_eq!(inflight.cache_tokens[workers[0].url()], 10 * 8192);
        assert_eq!(inflight.cache_tokens[workers[1].url()], 10 * 8192);
    }

    #[test]
    fn fallback_and_certified_concurrency_keep_both_ledgers_aligned() {
        let model = "kimi-k3";
        let tokens = StdArc::new(vec![9u32; 100]);
        let workers: StdArc<Vec<Arc<dyn Worker>>> = StdArc::new(vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ]);
        let policy = StdArc::new(cache_policy(
            LeastLoadCacheMode::Enforce,
            1000.0,
            1000.0,
            100,
        ));
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            workers.as_slice(),
            10,
            &[0, 1],
            None,
        )));
        let mut loads = HashMap::new();
        loads.insert(workers[0].url().to_string(), make_m2_load(0, 0, 1000.0));
        loads.insert(workers[1].url().to_string(), make_m2_load(0, 0, 1000.0));
        policy.update_loads(&loads);

        let start = StdArc::new(Barrier::new(17));
        std::thread::scope(|scope| {
            for thread_idx in 0..16 {
                let policy = StdArc::clone(&policy);
                let workers = StdArc::clone(&workers);
                let tokens = StdArc::clone(&tokens);
                let start = StdArc::clone(&start);
                scope.spawn(move || {
                    // The mismatched model takes the exact legacy fallback;
                    // the matching model takes the certified hybrid path.
                    let requested_model = if thread_idx % 2 == 0 { model } else { "other" };
                    start.wait();
                    for _ in 0..100 {
                        policy
                            .select_worker_cache_credit_chat(
                                workers.as_slice(),
                                &SelectWorkerInfo {
                                    tokens: Some(tokens.as_slice()),
                                    ..Default::default()
                                },
                                requested_model,
                            )
                            .unwrap();
                    }
                });
            }
            start.wait();
        });

        let inflight = policy.inflight.read().unwrap();
        for worker in workers.iter() {
            assert_eq!(
                inflight.cache_tokens[worker.url()],
                inflight.legacy_tokens[worker.url()],
                "legacy and bounded credits must be one atomic routing decision"
            );
        }
        let counts: Vec<_> = workers
            .iter()
            .map(|worker| inflight.legacy_tokens[worker.url()] / tokens.len() as u64)
            .collect();
        assert!(counts[0].abs_diff(counts[1]) <= 1);
    }

    #[test]
    fn load_poll_reset_cannot_split_a_routing_credit() {
        let model = "kimi-k3";
        let tokens = StdArc::new(vec![11u32; 100]);
        let workers: StdArc<Vec<Arc<dyn Worker>>> = StdArc::new(vec![
            mk_model("http://a:8000", model),
            mk_model("http://b:8000", model),
        ]);
        let policy = StdArc::new(cache_policy(
            LeastLoadCacheMode::Enforce,
            1000.0,
            1000.0,
            100,
        ));
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            workers.as_slice(),
            10,
            &[0, 1],
            None,
        )));
        let mut loads = HashMap::new();
        loads.insert(workers[0].url().to_string(), make_m2_load(0, 0, 1000.0));
        loads.insert(workers[1].url().to_string(), make_m2_load(0, 0, 1000.0));
        policy.update_loads(&loads);

        // Freeze the shared ledger. The poll takes the load write lock and
        // waits here, so a new route cannot observe the new snapshot until
        // both ledgers have been reset.
        let state_guard = policy.inflight.write().unwrap();
        let poll_policy = StdArc::clone(&policy);
        let poll_loads = loads.clone();
        let poll = std::thread::spawn(move || poll_policy.update_loads(&poll_loads));
        let mut poll_has_load_lock = false;
        for _ in 0..10_000 {
            if policy.cached_loads.try_read().is_err() {
                poll_has_load_lock = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(poll_has_load_lock, "poll did not reach the reset barrier");

        let (tx, rx) = mpsc::channel();
        let route_policy = StdArc::clone(&policy);
        let route_workers = StdArc::clone(&workers);
        let route_tokens = StdArc::clone(&tokens);
        let route = std::thread::spawn(move || {
            let selected = route_policy
                .select_worker_cache_credit_chat(
                    route_workers.as_slice(),
                    &SelectWorkerInfo {
                        tokens: Some(route_tokens.as_slice()),
                        ..Default::default()
                    },
                    model,
                )
                .unwrap();
            tx.send(selected).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(state_guard);
        poll.join().unwrap();
        let selected = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        route.join().unwrap();

        let inflight = policy.inflight.read().unwrap();
        assert_eq!(inflight.legacy_tokens[workers[selected].url()], 100);
        assert_eq!(inflight.cache_tokens[workers[selected].url()], 100);
        let other = 1 - selected;
        assert_eq!(inflight.legacy_tokens[workers[other].url()], 0);
        assert_eq!(inflight.cache_tokens[workers[other].url()], 0);
    }

    #[test]
    fn no_prefix_preserves_legacy_waterfill_across_all_workers() {
        let model = "kimi-k3";
        let tokens = vec![42u32; 128];
        let workers: Vec<_> = (0..51)
            .map(|idx| mk_model(&format!("http://w{idx}:8000"), model))
            .collect();
        let policy = StdArc::new(cache_policy(
            LeastLoadCacheMode::Enforce,
            322.0,
            8000.0,
            2048,
        ));
        policy.set_kv_event_monitor(Some(certified_monitor(
            model,
            &workers,
            16,
            &(0..51).collect::<Vec<_>>(),
            None,
        )));
        let mut loads = HashMap::new();
        for (idx, worker) in workers.iter().enumerate() {
            loads.insert(
                worker.url().to_string(),
                if idx == 0 {
                    // Exact M2 TokenSpeed shape: counts are populated, queued
                    // prompt tokens and generation throughput are both zero.
                    make_m2_load(38, 109, 0.0)
                } else {
                    make_m2_load(0, 0, 0.0)
                },
            );
        }
        policy.update_loads(&loads);

        let workers = StdArc::new(workers);
        let tokens = StdArc::new(tokens);
        let start = StdArc::new(Barrier::new(52));
        let selections = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..51 {
                let policy = StdArc::clone(&policy);
                let workers = StdArc::clone(&workers);
                let tokens = StdArc::clone(&tokens);
                let start = StdArc::clone(&start);
                handles.push(scope.spawn(move || {
                    start.wait();
                    (0..10)
                        .map(|_| {
                            policy
                                .select_worker_cache_credit_chat(
                                    workers.as_slice(),
                                    &SelectWorkerInfo {
                                        tokens: Some(tokens.as_slice()),
                                        ..Default::default()
                                    },
                                    model,
                                )
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                }));
            }
            start.wait();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(selections.len(), 510);
        let selected: HashSet<_> = selections.iter().copied().collect();
        assert_eq!(selected.len(), 51);
        let mut counts = vec![0usize; 51];
        for index in selections {
            counts[index] += 1;
        }
        assert!(counts.iter().all(|count| *count == 10));

        let inflight = policy.inflight.read().unwrap();
        assert!(inflight.cache_tokens.len() <= workers.len());
        assert_eq!(
            inflight
                .cache_tokens
                .values()
                .filter(|tokens| **tokens > 0)
                .count(),
            51
        );
        drop(inflight);
        policy.update_loads(&loads);
        assert!(policy
            .inflight
            .read()
            .unwrap()
            .cache_tokens
            .values()
            .all(|tokens| *tokens == 0));
        policy.remove_worker(workers[50].url());
        assert!(!policy
            .inflight
            .read()
            .unwrap()
            .cache_tokens
            .contains_key(workers[50].url()));
    }

    #[test]
    fn cold_start_picks_lowest_in_flight() {
        // No load reports yet -> join-shortest-queue on live in-flight count.
        let policy = LeastLoadPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            a.increment_load();
        }
        let workers = vec![a, b];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn routes_to_lower_queued_token_work() {
        // Equal KV/throughput; the worker with fewer queued tokens wins.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(8000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        policy.update_loads(&loads);
        // a: 8000/100 = 80s ; b: 1000/100 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn throughput_normalization_prefers_faster_worker() {
        // Same queued tokens; the faster worker (higher throughput) drains sooner.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(5000, 0.2, 50.0));
        loads.insert("http://b:8000".to_string(), make_load(5000, 0.2, 500.0));
        policy.update_loads(&loads);
        // a: 5000/50 = 100s ; b: 5000/500 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn zero_throughput_falls_back_to_default() {
        // A backend that reports no gen_throughput (0); the score must still
        // discriminate via the configured default_throughput, not collapse.
        let policy = LeastLoadPolicy::new(); // default_throughput = 2000
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(10000, 0.2, 0.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 0.0));
        policy.update_loads(&loads);
        // gen_throughput=0 -> default 2000: a 10000/2000=5s ; b 1000/2000=0.5s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn missing_snapshot_estimated_in_time_units() {
        // Worker a reports ~40s of queued work; worker b has no snapshot but 5
        // live in-flight. Scoring b on raw count (5) would wrongly beat a's 40s;
        // scoring it as drain time (5 * p̄ / nominal ≈ 51s) keeps the lighter a.
        let policy = LeastLoadPolicy::new(); // p̄ = 1024
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(4000, 0.0, 100.0));
        policy.update_loads(&loads);
        // a: 4000/100 = 40s ; b: 5 * 1024 / 100 ≈ 51.2s -> pick a.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn kv_barrier_avoids_full_worker() {
        // No queued work; the convex KV barrier steers off the near-full worker.
        let policy = LeastLoadPolicy::with_kv_pressure_weight(2.0);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.98, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.0, 100.0));
        policy.update_loads(&loads);
        // a: 0 + 2*0.98/0.02 = 98 ; b: 0 -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn inflight_correction_spreads_within_poll_interval() {
        // Two identical workers, no fresh poll between dispatches: the in-flight
        // token credit must push the second request to the other worker rather
        // than herding both onto the first.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        let info = SelectWorkerInfo::default(); // tokens unknown -> mean prefill
        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn update_loads_resets_inflight() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        let info = SelectWorkerInfo::default();
        for _ in 0..4 {
            policy.select_worker(&workers, &info);
        }
        assert!(policy
            .inflight
            .read()
            .unwrap()
            .legacy_tokens
            .values()
            .any(|&v| v > 0));

        // A fresh poll clears the since-poll estimate.
        policy.update_loads(&loads);
        assert!(policy
            .inflight
            .read()
            .unwrap()
            .legacy_tokens
            .values()
            .all(|&v| v == 0));
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000")];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn remove_worker_prunes_state() {
        let policy = LeastLoadPolicy::new();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.5, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.3, 100.0));
        policy.update_loads(&loads);
        assert_eq!(policy.cached_loads.read().unwrap().len(), 2);

        // Removing a worker drops only its entry (no unbounded growth on churn).
        policy.remove_worker("http://a:8000");
        let cached = policy.cached_loads.read().unwrap();
        assert_eq!(cached.len(), 1);
        assert!(!cached.contains_key("http://a:8000"));
        assert!(cached.contains_key("http://b:8000"));
    }
}
