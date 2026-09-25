use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    BucketPolicy, CacheAwarePolicy, DPRankLoadPolicy, LeastLoadPolicy, LoadBalancingPolicy,
    ManualConfig, ManualPolicy, OwnerPressureDispatchPlan, PolicyFactory, SeedWorkerHeadroom,
    SelectWorkerInfo,
};
use crate::{
    config::types::{PolicyConfig, RoutingKeyOverrideConfig},
    policies::cache_aware::LoadReceiver,
    routers::common::header_utils::extract_routing_key,
    worker::{KvEventMonitor, Worker},
};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>>>,

    /// Model ID -> Worker count for cleanup tracking (lock-free reads via DashMap)
    model_worker_counts: Arc<DashMap<String, usize>>,

    /// Default policy instance (cached, immutable after creation)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Explicit model policy configurations supplied at router startup.
    model_policy_configs: Arc<HashMap<String, PolicyConfig>>,

    /// Prefill policy for PD mode (set once at startup, lock-free reads via OnceLock)
    prefill_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Decode policy for PD mode (set once at startup, lock-free reads via OnceLock)
    decode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Encode policy for EPD mode (set once at startup, lock-free reads via OnceLock)
    encode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Optional KV event monitor for policies that consume backend cache events.
    /// When set, new eligible policy instances are injected with this monitor.
    kv_event_monitor: Arc<RwLock<Option<Arc<KvEventMonitor>>>>,

    /// Optional backend load-snapshot receiver from the `WorkerMonitor`. When
    /// set, new CacheAwarePolicy instances are injected with it for the KV-usage
    /// imbalance trigger.
    load_rx: Arc<RwLock<Option<LoadReceiver>>>,

    // DP-rank policy: Supports the selection of dp-rank outside the engine.
    dp_rank_policy: Arc<OnceLock<Arc<dyn DPRankLoadPolicy>>>,

    /// Shared sticky selector for the `X-SMG-Routing-Key` override. `Some` when the
    /// override is enabled; consulted (instead of the configured policy) for keyed
    /// requests via [`PolicyRegistry::select_worker`].
    routing_key_sticky: Option<Arc<ManualPolicy>>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy (no routing-key override).
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        Self::with_override(default_policy_config, RoutingKeyOverrideConfig::default())
    }

    /// Create a PolicyRegistry. When `routing_key_override.enabled`, builds a shared
    /// sticky selector consulted for keyed requests in [`Self::select_worker`].
    pub fn with_override(
        default_policy_config: PolicyConfig,
        routing_key_override: RoutingKeyOverrideConfig,
    ) -> Self {
        Self::with_model_policies(default_policy_config, routing_key_override, HashMap::new())
    }

    /// Create a registry with explicit per-model policy overrides.
    pub fn with_model_policies(
        default_policy_config: PolicyConfig,
        routing_key_override: RoutingKeyOverrideConfig,
        model_policy_configs: HashMap<String, PolicyConfig>,
    ) -> Self {
        let default_policy = Self::create_policy_from_config(&default_policy_config);
        let routing_key_sticky = routing_key_override.enabled.then(|| {
            Arc::new(ManualPolicy::with_config(ManualConfig {
                eviction_interval_secs: routing_key_override.eviction_interval_secs,
                max_idle_secs: routing_key_override.max_idle_secs,
                assignment_mode: routing_key_override.assignment_mode,
            }))
        });

        Self {
            model_policies: Arc::new(DashMap::new()),
            model_worker_counts: Arc::new(DashMap::new()),
            default_policy,
            model_policy_configs: Arc::new(model_policy_configs),
            prefill_policy: Arc::new(OnceLock::new()),
            decode_policy: Arc::new(OnceLock::new()),
            encode_policy: Arc::new(OnceLock::new()),
            kv_event_monitor: Arc::new(RwLock::new(None)),
            load_rx: Arc::new(RwLock::new(None)),
            dp_rank_policy: Arc::new(OnceLock::new()),
            routing_key_sticky,
        }
    }

    /// Select a worker, applying the `X-SMG-Routing-Key` sticky override when it is
    /// enabled, the request carries the header, and the configured policy does not
    /// already honor the key (`manual` / `consistent_hashing`). Otherwise delegates
    /// to `policy`. `policy.name()` stays the real policy (for metrics).
    pub fn select_worker(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name())
                && extract_routing_key(info.headers).is_some()
            {
                return sticky.select_worker(workers, info);
            }
        }
        policy.select_worker(workers, info)
    }

    /// Select a worker and report any router-local work reserved by the policy.
    ///
    /// The sticky routing-key override does not reserve work in `policy`, so it
    /// must return no reservation. Callers use the returned cost to attach a
    /// request-lifetime release guard to the selected worker.
    pub fn select_worker_with_reservation(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<(usize, Option<u64>)> {
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name())
                && extract_routing_key(info.headers).is_some()
            {
                return sticky.select_worker(workers, info).map(|idx| (idx, None));
            }
        }

        let idx = policy.select_worker(workers, info)?;
        Some((idx, policy.reservation_cost(info)))
    }

    /// Narrow least-load cache-credit entry point. The gRPC worker-selection
    /// stage calls this only for direct Regular scalar streaming Chat.
    pub fn select_worker_with_reservation_cache_credit_chat(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        model_id: &str,
    ) -> Option<(usize, Option<u64>)> {
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name())
                && extract_routing_key(info.headers).is_some()
            {
                return sticky.select_worker(workers, info).map(|idx| (idx, None));
            }
        }

        let idx = if let Some(least_load) = policy.as_any().downcast_ref::<LeastLoadPolicy>() {
            least_load.select_worker_cache_credit_chat(workers, info, model_id)?
        } else {
            policy.select_worker(workers, info)?
        };
        Some((idx, policy.reservation_cost(info)))
    }

    /// Policies that already honor `X-SMG-Routing-Key` keep their own handling; all
    /// others (cache_aware, least_load, prefix_hash, ...) get the sticky override.
    fn routing_key_override_applies(name: &str) -> bool {
        !matches!(name, "manual" | "consistent_hashing")
    }

    /// Set the KV event monitor (thread-safe, can be called after initialization).
    /// Propagates to every existing policy that consumes KV events.
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        {
            let mut guard = self.kv_event_monitor.write();
            guard.clone_from(&monitor);
        }

        // Propagate to existing KV-event consumers so they don't miss the monitor.
        // This covers the default_policy (created before the monitor was available)
        // and any model/PD policies that were already set up.
        Self::maybe_inject_monitor(&self.default_policy, monitor.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        if let Some(p) = self.encode_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_monitor(entry.value(), monitor.as_ref());
        }
    }

    /// Inject KV event monitor into a policy if it consumes KV events.
    fn maybe_inject_monitor(
        policy: &Arc<dyn LoadBalancingPolicy>,
        monitor: Option<&Arc<KvEventMonitor>>,
    ) {
        if policy.needs_kv_events() {
            policy.set_kv_event_monitor(monitor.cloned());
        }
    }

    /// Set the backend load-snapshot receiver (thread-safe, can be called after
    /// initialization). Propagates to all existing cache-aware policies.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        {
            let mut guard = self.load_rx.write();
            guard.clone_from(&rx);
        }
        Self::maybe_inject_load_rx(&self.default_policy, rx.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        if let Some(p) = self.encode_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_load_rx(entry.value(), rx.as_ref());
        }
    }

    /// Inject the load receiver into a policy if it's cache-aware.
    fn maybe_inject_load_rx(policy: &Arc<dyn LoadBalancingPolicy>, rx: Option<&LoadReceiver>) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_load_receiver(rx.cloned());
        }
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count using DashMap entry API
        let count = self
            .model_worker_counts
            .entry(model_id.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        debug!("Worker added for model {}, count: {}", model_id, *count);
        drop(count); // Release the entry lock

        // Check if model already has a policy (lock-free read via DashMap)
        if let Some(existing_policy) = self.model_policies.get(model_id) {
            debug!(
                "Model {} already has policy: {}",
                model_id,
                existing_policy.name()
            );
            return Arc::clone(&existing_policy);
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Store policy for this model (DashMap handles concurrent inserts)
        self.model_policies
            .insert(model_id.to_string(), Arc::clone(&policy));

        policy
    }

    /// Called when a worker is removed
    pub fn on_worker_removed(&self, model_id: &str) {
        // Decrement worker count and check if cleanup needed
        let should_cleanup = if let Some(mut count_ref) = self.model_worker_counts.get_mut(model_id)
        {
            *count_ref = count_ref.saturating_sub(1);
            debug!(
                "Worker removed for model {}, count: {}",
                model_id, *count_ref
            );
            if *count_ref == 0 {
                drop(count_ref); // Release before remove
                self.model_worker_counts.remove(model_id);
                true
            } else {
                false
            }
        } else {
            warn!(
                "Attempted to remove worker for model {} with no registered workers",
                model_id
            );
            false
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            if let Some((_, policy)) = self.model_policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
            }
        }
    }

    /// Get the policy for a model (lock-free via DashMap)
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Ask only the active cache-aware policy for a read-only clean-peer plan.
    /// Other policies fail closed, so a scheduler proof can never become a
    /// policy-agnostic adaptive-admission bypass.
    pub(crate) fn owner_pressure_dispatch_plan(
        &self,
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        headroom: &[SeedWorkerHeadroom],
    ) -> Option<OwnerPressureDispatchPlan> {
        let policy = self.get_policy_or_default(model_id);
        policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()?
            .owner_pressure_dispatch_plan(model_id, workers, info, headroom)
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Explicit router configuration wins over worker-supplied labels.
        if let Some(config) = self.model_policy_configs.get(model_id) {
            debug!(
                "Using configured policy '{}' for model {}",
                config.name(),
                model_id
            );
            return self.create_configured_policy(config);
        }

        // 2. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 3. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string (delegates to PolicyFactory)
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        if policy_type == "cache_aware" {
            let cache_aware = CacheAwarePolicy::new();
            {
                let guard = self.kv_event_monitor.read();
                if let Some(ref monitor) = *guard {
                    cache_aware.set_kv_event_monitor(Some(Arc::clone(monitor)));
                }
            }
            {
                let guard = self.load_rx.read();
                if let Some(ref rx) = *guard {
                    cache_aware.set_load_receiver(Some(rx.clone()));
                }
            }
            Arc::new(cache_aware)
        } else {
            PolicyFactory::create_by_name(policy_type).unwrap_or_else(|| {
                warn!("Unknown policy type '{}', using default", policy_type);
                Arc::clone(&self.default_policy)
            })
        }
    }

    fn create_configured_policy(&self, config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        let policy = Self::create_policy_from_config(config);
        {
            let guard = self.kv_event_monitor.read();
            Self::maybe_inject_monitor(&policy, guard.as_ref());
        }
        {
            let guard = self.load_rx.read();
            Self::maybe_inject_load_rx(&policy, guard.as_ref());
        }
        policy
    }

    /// Create a policy from a PolicyConfig (delegates to PolicyFactory)
    fn create_policy_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        PolicyFactory::create_from_config(config)
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> HashMap<String, String> {
        self.model_policies
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> HashMap<String, usize> {
        self.model_worker_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        self.model_policies.clear();
        self.model_worker_counts.clear();
    }

    /// Set the prefill policy for PD mode (lock-free, set once at startup)
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.prefill_policy.set(policy);
    }

    pub fn set_dp_rank_policy(&self, policy: Arc<dyn DPRankLoadPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        debug!("set dp rank policy");
        let _ = self.dp_rank_policy.set(policy);
    }

    pub fn get_dp_rank_policy(&self) -> Option<Arc<dyn DPRankLoadPolicy>> {
        self.dp_rank_policy.get().map(Arc::clone)
    }

    /// Set the decode policy for PD mode (lock-free, set once at startup)
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.decode_policy.set(policy);
    }

    /// Set the encode policy for EPD mode (lock-free, set once at startup)
    pub fn set_encode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.encode_policy.set(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set (lock-free)
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.prefill_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set (lock-free)
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.decode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the encode policy for EPD mode. Falls back to consistent_hashing so
    /// repeated multimodal items keep stable affinity even when the main policy is
    /// load-oriented or random.
    pub fn get_encode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.encode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing))
    }

    /// Get all load-aware policies that need periodic load updates (lock-free).
    pub fn get_all_load_aware_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut policies = Vec::new();

        if self.default_policy.needs_load_updates() {
            policies.push(Arc::clone(&self.default_policy));
        }

        // Get prefill, decode, and encode policies (lock-free via OnceLock::get)
        let prefill_policy_opt = self.prefill_policy.get();
        let decode_policy_opt = self.decode_policy.get();
        let encode_policy_opt = self.encode_policy.get();

        if let Some(policy) = prefill_policy_opt {
            if policy.needs_load_updates() && !Arc::ptr_eq(policy, &self.default_policy) {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = decode_policy_opt {
            if policy.needs_load_updates()
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = encode_policy_opt {
            if policy.needs_load_updates()
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
                && !decode_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        for entry in self.model_policies.iter() {
            let policy = entry.value();
            if policy.needs_load_updates() {
                let already_added = policies.iter().any(|p| Arc::ptr_eq(p, policy));
                if !already_added {
                    policies.push(Arc::clone(policy));
                }
            }
        }

        policies
    }

    /// Get all PowerOfTwo policies that need load updates (lock-free).
    ///
    /// Kept for compatibility with callers that only want the original
    /// PowerOfTwo subset.
    pub fn get_all_power_of_two_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        self.get_all_load_aware_policies()
            .into_iter()
            .filter(|policy| policy.name() == "power_of_two")
            .collect()
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    debug!(
                        "Initializing cache-aware policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cache_aware.init_workers(workers);
                }
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_worker_from_model(model_id, worker_url);
                    debug!(
                        "Removed worker {} from cache-aware policy for model {}",
                        worker_url, model_id
                    );
                }
            }
        }
    }

    /// Remove a worker from PD cache-aware policies if applicable
    /// This should be called when a prefill or decode worker is being removed
    pub fn remove_worker_from_pd_cache_aware(&self, worker_url: &str) {
        for (worker_type, policy) in [
            ("prefill", self.prefill_policy.get()),
            ("decode", self.decode_policy.get()),
            ("encode", self.encode_policy.get()),
        ] {
            if let Some(policy) = policy {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                        cache_aware.remove_worker_by_url(worker_url);
                        debug!(
                            "Removed worker {} from {} cache-aware policy",
                            worker_url, worker_type
                        );
                    }
                }
            }
        }
    }

    /// Drop a removed worker's cached load report from all load-aware policies
    /// (`power_of_two`, `least_load`).
    ///
    /// These policies cache per-worker load reports keyed by URL; without this
    /// their caches would grow unbounded under worker churn. Called on worker
    /// removal alongside the cache-aware cleanup above.
    pub fn remove_worker_from_load_aware(&self, worker_url: &str) {
        for policy in self.get_all_load_aware_policies() {
            policy.remove_worker(worker_url);
        }
    }

    /// Initialize cache-aware policies for PD mode (prefill and decode) - lock-free
    pub fn init_pd_cache_aware_policies(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) {
        // Initialize prefill policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache-aware policy with {} workers",
                            prefill_workers.len()
                        );
                        cache_aware.init_workers(prefill_workers);
                    }
                }
            }
        }

        // Initialize decode policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache-aware policy with {} workers",
                            decode_workers.len()
                        );
                        cache_aware.init_workers(decode_workers);
                    }
                }
            }
        }
    }

    /// Initialize bucket policies for PD mode - lock-free
    pub fn init_pd_bucket_policies(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Initialize prefill policy if it's bucket (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "bucket" {
                if let Some(bucket) = prefill_policy.as_any().downcast_ref::<BucketPolicy>() {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill bucket policy with {} workers",
                            prefill_workers.len()
                        );
                        bucket.init_prefill_worker_urls(prefill_workers);
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        policies::{CacheAwareConfig, SelectWorkerInfo},
        worker::{BasicWorkerBuilder, Worker, WorkerType},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn cache_aware_policy() -> Arc<dyn LoadBalancingPolicy> {
        Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        }))
    }

    fn headers_with_key(key: &str) -> http::HeaderMap {
        let mut h = http::HeaderMap::new();
        h.insert("x-smg-routing-key", key.parse().unwrap());
        h
    }

    #[test]
    fn override_eligibility_skips_key_native_policies() {
        // Policies that already honor X-SMG-Routing-Key are skipped; others (incl.
        // prefix_hash, which routes by tokens) get the sticky override.
        assert!(PolicyRegistry::routing_key_override_applies("cache_aware"));
        assert!(PolicyRegistry::routing_key_override_applies("prefix_hash"));
        assert!(PolicyRegistry::routing_key_override_applies("least_load"));
        assert!(!PolicyRegistry::routing_key_override_applies("manual"));
        assert!(!PolicyRegistry::routing_key_override_applies(
            "consistent_hashing"
        ));
    }

    #[test]
    fn override_routes_keyed_request_stickily() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
            worker("http://w3", WorkerType::Regular),
        ];
        let headers = headers_with_key("session-A");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let first = reg.select_worker(&policy, &workers, &info).unwrap();
        for _ in 0..5 {
            assert_eq!(reg.select_worker(&policy, &workers, &info), Some(first));
        }
    }

    #[test]
    fn override_without_key_uses_configured_policy() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo::default(); // no key header
                                                // RoundRobin alternates -> proves the configured policy is used, not sticky.
        let a = reg.select_worker(&policy, &workers, &info).unwrap();
        let b = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn override_disabled_ignores_key() {
        let reg = PolicyRegistry::new(PolicyConfig::RoundRobin); // override off
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let headers = headers_with_key("session-A");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        // Override off -> the key is ignored, RoundRobin alternates.
        let a = reg.select_worker(&policy, &workers, &info).unwrap();
        let b = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[test]
    fn test_passthrough_is_not_load_aware() {
        // Passthrough must not be polled by the WorkerMonitor: with it as the
        // default policy (and a passthrough model policy too), the load-aware
        // set stays empty.
        let registry = PolicyRegistry::new(PolicyConfig::Passthrough);
        registry.on_worker_added("m", Some("passthrough"));
        assert!(registry.get_all_load_aware_policies().is_empty());
    }

    #[test]
    fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }

    #[test]
    fn test_size_aware_default_applies_to_every_model_without_a_hint() {
        let registry = PolicyRegistry::new(PolicyConfig::SizeAwarePowerOfTwo {
            output_token_estimate: 2048,
        });

        let model_a = registry.on_worker_added("model-a", None);
        let model_b = registry.on_worker_added("model-b", None);

        assert_eq!(model_a.name(), "size_aware_power_of_two");
        assert_eq!(model_b.name(), "size_aware_power_of_two");
        assert!(Arc::ptr_eq(&model_a, &model_b));
    }

    #[test]
    fn select_worker_with_reservation_reports_atomic_size_aware_work() {
        let registry = PolicyRegistry::new(PolicyConfig::SizeAwarePowerOfTwo {
            output_token_estimate: 2048,
        });
        let policy = registry.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let tokens = vec![0; 100];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            max_output_tokens: Some(500),
            reserve_work: true,
            ..Default::default()
        };

        let (first, first_cost) = registry
            .select_worker_with_reservation(&policy, &workers, &info)
            .unwrap();
        let (second, second_cost) = registry
            .select_worker_with_reservation(&policy, &workers, &info)
            .unwrap();

        assert_ne!(
            first, second,
            "the second admission must see the first reservation"
        );
        assert_eq!(first_cost, Some(600));
        assert_eq!(second_cost, Some(600));
        policy.release_reservation(workers[first].url(), first_cost.unwrap());
        policy.release_reservation(workers[second].url(), second_cost.unwrap());
    }

    #[test]
    fn sticky_override_does_not_report_an_unmade_policy_reservation() {
        let registry = PolicyRegistry::with_override(
            PolicyConfig::SizeAwarePowerOfTwo {
                output_token_estimate: 2048,
            },
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = registry.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let headers = headers_with_key("session-A");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            reserve_work: true,
            ..Default::default()
        };

        let (_, reservation) = registry
            .select_worker_with_reservation(&policy, &workers, &info)
            .unwrap();
        assert_eq!(reservation, None);
    }

    #[test]
    fn test_explicit_model_policy_overrides_default_and_worker_hint() {
        let registry = PolicyRegistry::with_model_policies(
            PolicyConfig::SizeAwarePowerOfTwo {
                output_token_estimate: 2048,
            },
            RoutingKeyOverrideConfig::default(),
            HashMap::from([(
                "kimi-k3".to_string(),
                PolicyConfig::CacheAware {
                    cache_threshold: 0.0,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 30,
                    max_tree_size: 1_000_000,
                    fallback_output_token_estimate: 4096,
                    block_size: 16,
                    engine_load: true,
                    balance_token_usage_threshold: 1.0,
                    overload_token_usage_threshold: 1.0,
                    max_cached_owners_per_prefix: 0,
                    cache_owner_spill_cooldown_secs: 0,
                },
            )]),
        );

        let k3 = registry.on_worker_added("kimi-k3", Some("round_robin"));
        let other = registry.on_worker_added("other-model", None);

        assert_eq!(k3.name(), "cache_aware");
        assert_eq!(other.name(), "size_aware_power_of_two");
    }

    #[test]
    fn test_pd_cache_aware_policy_initialization() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_prefill_policy(cache_aware_policy());
        registry.set_decode_policy(cache_aware_policy());

        let prefill_workers = vec![
            worker("http://prefill-1:8000", WorkerType::Prefill),
            worker("http://prefill-2:8000", WorkerType::Prefill),
        ];
        let decode_workers = vec![
            worker("http://decode-1:8000", WorkerType::Decode),
            worker("http://decode-2:8000", WorkerType::Decode),
        ];

        registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        let prefill_policy = registry.get_prefill_policy();
        let decode_policy = registry.get_decode_policy();
        let info = SelectWorkerInfo {
            request_text: Some("shared prefix request"),
            ..Default::default()
        };

        let prefill_first = prefill_policy.select_worker(&prefill_workers, &info);
        let prefill_second = prefill_policy.select_worker(&prefill_workers, &info);
        assert!(prefill_first.is_some());
        assert_eq!(prefill_first, prefill_second);

        let decode_first = decode_policy.select_worker(&decode_workers, &info);
        let decode_second = decode_policy.select_worker(&decode_workers, &info);
        assert!(decode_first.is_some());
        assert_eq!(decode_first, decode_second);

        registry.remove_worker_from_pd_cache_aware("http://prefill-1:8000");
        registry.remove_worker_from_pd_cache_aware("http://decode-1:8000");
    }
}
