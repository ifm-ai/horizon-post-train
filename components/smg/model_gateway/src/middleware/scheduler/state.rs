//! Startup wiring for the priority scheduler: builds the admission mode
//! the route layer branches on.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use axum::http::HeaderMap;
use tokio::sync::{broadcast, watch};
use tracing::{error, info};

use super::{
    capacity_credit::{CapacityCreditRegistry, HeldSchedulerPermit},
    fair_share::{FairShareProfile, REQUEST_MODEL_HEADER},
    Class, GlobalFairShare, PriorityScheduler, SchedulerSettings, StaticTenantPolicyResolver,
    TenantPolicyResolver,
};
use crate::{
    config::types::RouterConfig,
    middleware::token_bucket::TokenBucket,
    observability::metrics::Metrics,
    worker::{event::WorkerEvent, CapacityTrackerSettings, WorkerCapacity, WorkerRegistry},
};

/// How often the metrics sampler refreshes the capacity / autoscaling gauges.
const SAMPLER_INTERVAL: Duration = Duration::from_secs(5);

/// Trusted selector injected by the Comet proxy after it strips any
/// client-supplied value. Ordinary requests carry their model id; requests
/// routed to a private reservation carry `private`.
pub const ADMISSION_PARTITION_HEADER: &str = "x-smg-admission-partition";

/// Worker metadata label used by trusted control planes to assign a healthy
/// replica to a partition other than its primary model id.
pub const ADMISSION_PARTITION_LABEL: &str = "admission_partition";

/// Optional live total-capacity ceiling supplied by adaptive admission.
/// SlotPool remains authoritative for local in-flight accounting.
pub(crate) trait AdaptiveCapacityProvider: Send + Sync {
    fn effective_capacity(&self, partition: &str, static_capacity: u16) -> u16;

    fn subscribe_capacity_changes(&self) -> watch::Receiver<u64>;
}

#[derive(Clone)]
pub struct SchedulerPartition {
    pub name: Arc<str>,
    pub scheduler: Arc<PriorityScheduler>,
}

/// State handed to `priority_admission_middleware` via `from_fn_with_state`.
/// Cheap to clone (all `Arc`).
pub struct SchedulerState {
    /// Default scheduler, retained as a direct field for the original
    /// unpartitioned API and tests.
    pub scheduler: Arc<PriorityScheduler>,
    partitions: HashMap<String, SchedulerPartition>,
    default_partition: Arc<str>,
    pub resolver: Arc<dyn TenantPolicyResolver>,
    model_registry: Arc<WorkerRegistry>,
    /// Per-second RPS sibling check, run before admission. Set only when an
    /// explicit `rate_limit_tokens_per_second` is configured; the bucket's
    /// concurrency-cap role is owned by the scheduler, so we must not consult
    /// it as a concurrency limiter (that would double-limit). `None` =
    /// no RPS limit.
    pub rate_limiter: Option<Arc<TokenBucket>>,
    capacity_credits: Option<Arc<CapacityCreditRegistry<HeldSchedulerPermit>>>,
    capacity_credit_required: bool,
}

impl SchedulerState {
    /// Select an admission partition using only a trusted, exact header
    /// match. Missing, malformed, and unknown selectors use the configured
    /// default partition.
    pub fn partition_for(&self, headers: &HeaderMap) -> SchedulerPartition {
        let requested = headers
            .get(ADMISSION_PARTITION_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        requested
            .and_then(|name| self.partitions.get(name))
            .cloned()
            .unwrap_or_else(|| SchedulerPartition {
                name: Arc::clone(&self.default_partition),
                scheduler: Arc::clone(&self.scheduler),
            })
    }

    /// Resolve an exact configured partition. Capacity credits never fall
    /// back because their binding must identify the same non-fungible pool at
    /// issue and redemption time.
    pub(crate) fn partition_exact(&self, name: &str) -> Option<SchedulerPartition> {
        self.partitions.get(name).cloned()
    }

    pub(crate) fn capacity_credits(
        &self,
    ) -> Option<&Arc<CapacityCreditRegistry<HeldSchedulerPermit>>> {
        self.capacity_credits.as_ref()
    }

    #[must_use]
    pub(crate) fn capacity_credit_required(&self) -> bool {
        self.capacity_credit_required
    }

    pub(crate) fn canonical_model(&self, raw_model: &str) -> Arc<str> {
        self.model_registry
            .resolve_model_alias(raw_model)
            .unwrap_or_else(|| Arc::from(raw_model))
    }

    /// Resolve the trusted request model to a configured fair-share profile.
    /// Model aliases are canonicalized only for accounting; the outbound
    /// request body remains untouched by the scheduler.
    pub(crate) fn fair_share_profile_for(
        &self,
        headers: &HeaderMap,
        ledger: &GlobalFairShare,
    ) -> FairShareProfile {
        if !ledger.has_model_profiles() {
            return FairShareProfile::Global;
        }
        if !ledger.trusts_request_model_header() {
            super::metrics::record_fair_share_fallback("untrusted_request_model");
            return FairShareProfile::Global;
        }
        let Some(raw_model) = headers
            .get(REQUEST_MODEL_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            super::metrics::record_fair_share_fallback("missing_request_model");
            return FairShareProfile::Global;
        };
        let canonical = self.canonical_model(raw_model);
        ledger.profile_for_model(&canonical)
    }

    pub(crate) fn fair_share_profile_for_model(
        &self,
        raw_model: &str,
        ledger: &GlobalFairShare,
    ) -> FairShareProfile {
        if !ledger.has_model_profiles() {
            return FairShareProfile::Global;
        }
        ledger.profile_for_model(&self.canonical_model(raw_model))
    }
}

/// Which admission path the protected routes use. Chosen once at startup.
#[derive(Clone)]
pub enum AdmissionMode {
    /// Legacy `concurrency_limit_middleware` (default; zero behavior change).
    Legacy,
    /// Priority scheduler enabled.
    Priority(Arc<SchedulerState>),
}

impl AdmissionMode {
    /// Build the admission mode from runtime config.
    ///
    /// When `priority_scheduler_enabled` is false, returns `Legacy` without
    /// constructing anything. When true, constructs `WorkerCapacity` over
    /// the worker fleet, builds the scheduler against its current capacity,
    /// spawns the dispatcher on its watch channel, and returns
    /// `Priority(..)`.
    ///
    /// On any startup error (bad YAML, reservations exceed capacity), logs
    /// at ERROR and falls back to `Legacy` rather than aborting the whole
    /// gateway — a misconfigured scheduler must not take the data plane down.
    pub fn from_config(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
    ) -> Self {
        match Self::try_from_config_with_adaptive(rc, registry, rate_limiter, None) {
            Ok(mode) => mode,
            Err(e) => {
                error!(
                    error = %e,
                    "priority scheduler failed to start; falling back to legacy admission"
                );
                Self::Legacy
            }
        }
    }

    /// Startup path used by the production server.
    ///
    /// Existing scheduler-only deployments preserve the legacy fail-safe
    /// fallback. Once capacity credits or adaptive scheduler capacity are
    /// configured, construction errors are returned to the caller because the
    /// admission boundary must never silently disappear.
    pub(crate) fn try_from_config_with_adaptive(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
        adaptive_capacity_provider: Option<Arc<dyn AdaptiveCapacityProvider>>,
    ) -> Result<Self, String> {
        if !rc.priority_scheduler_enabled {
            return Ok(Self::Legacy);
        }
        match Self::try_build_priority(rc, registry, rate_limiter, adaptive_capacity_provider) {
            Ok(mode) => {
                info!("priority scheduler enabled");
                Ok(mode)
            }
            Err(error)
                if rc.capacity_credit_generation.is_none()
                    && !rc.capacity_credit_required
                    && !rc.priority_scheduler_adaptive_capacity =>
            {
                tracing::error!(
                    error = %error,
                    "priority scheduler failed to start; falling back to legacy admission"
                );
                Ok(Self::Legacy)
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) fn try_build_priority(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
        adaptive_capacity_provider: Option<Arc<dyn AdaptiveCapacityProvider>>,
    ) -> Result<Self, String> {
        // The configured concurrency value is one global ceiling across the
        // entire healthy worker fleet. Worker-reported capacity may lower the
        // scheduler limit, but it must never raise it above this contract.
        let configured_max = if rc.max_concurrent_requests > 0 {
            Some(u16::try_from(rc.max_concurrent_requests).unwrap_or(u16::MAX))
        } else {
            None
        };
        let cap_settings = CapacityTrackerSettings {
            max_capacity: configured_max,
            legacy_max_concurrent_requests: configured_max.unwrap_or_else(|| {
                CapacityTrackerSettings::default().legacy_max_concurrent_requests
            }),
            ..CapacityTrackerSettings::default()
        };
        let worker_capacity = WorkerCapacity::spawn(Arc::clone(&registry), cap_settings);
        // Keep a receiver alive as soon as the tracker exists. Tokio's
        // `watch::Sender::send` does not retain a value when no receiver is
        // subscribed, so parsing the scheduler configuration must not create
        // a gap where the first fleet update is lost.
        let capacity_watch = worker_capacity.watch();

        let default_max_class = Class::parse_header(&rc.priority_scheduler_default_max_class);
        let yaml = load_yaml(rc.priority_scheduler_config.as_deref())?;
        let mut settings = SchedulerSettings::from_cli_and_yaml(
            true,
            default_max_class,
            rc.priority_scheduler_tenant_metric_top_n,
            yaml.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        if yaml.is_none() {
            settings = settings.with_global_queue_budget(rc.queue_size);
        }
        let fair_share = GlobalFairShare::from_settings(&settings).map(Arc::new);
        let capacity_credits = if let Some(generation) = &rc.capacity_credit_generation {
            let ledger = fair_share.as_ref().ok_or_else(|| {
                "capacity credits require fair_share in the priority-scheduler config".to_string()
            })?;
            if !ledger.trusts_output_token_estimate_header()
                || !ledger.trusts_request_model_header()
            {
                return Err(
                    "capacity credits require trusted output-token estimates and request-model headers"
                        .to_string(),
                );
            }
            let registry = Arc::new(
                CapacityCreditRegistry::new(
                    generation,
                    Duration::from_millis(rc.capacity_credit_ttl_ms),
                    Duration::from_secs(rc.capacity_credit_terminal_retention_secs),
                )
                .map_err(|error| error.to_string())?,
            );
            registry.spawn_reaper(Duration::from_millis(
                rc.capacity_credit_ttl_ms.clamp(10, 1_000),
            ));
            Some(registry)
        } else {
            None
        };
        let adaptive_capacity_provider = if rc.priority_scheduler_adaptive_capacity {
            Some(adaptive_capacity_provider.ok_or_else(|| {
                "priority_scheduler_adaptive_capacity requires a live adaptive provider".to_string()
            })?)
        } else {
            None
        };
        let model_registry = Arc::clone(&registry);

        let resolver: Arc<dyn TenantPolicyResolver> =
            Arc::new(StaticTenantPolicyResolver::from_settings(&settings));

        // The scheduler owns concurrency, so the shared bucket only survives
        // as an RPS check when an explicit per-second limit is configured.
        let rate_limiter = match rc.rate_limit_tokens_per_second {
            Some(rps) if rps > 0 => rate_limiter,
            _ => None,
        };

        let Some(partition_config) = yaml
            .as_ref()
            .filter(|yaml| !yaml.admission_partitions.is_empty())
        else {
            if capacity_credits.is_some() || adaptive_capacity_provider.is_some() {
                return Err(
                    "capacity credits and adaptive scheduler capacity require explicit admission_partitions"
                        .to_string(),
                );
            }
            // The atomic value covers any update that won the race before the
            // receiver subscribed; subsequent updates remain queued for the
            // dispatcher through `capacity_watch`.
            let scheduler = PriorityScheduler::new_with_fair_share(
                &settings,
                worker_capacity.current(),
                fair_share,
            )
            .map_err(|e| e.to_string())?;
            scheduler.spawn_dispatcher_retaining_capacity(capacity_watch, worker_capacity);
            scheduler.spawn_sampler(SAMPLER_INTERVAL);
            return Ok(Self::Priority(Arc::new(SchedulerState {
                scheduler,
                partitions: HashMap::new(),
                default_partition: Arc::from("global"),
                resolver,
                model_registry,
                rate_limiter,
                capacity_credits: None,
                capacity_credit_required: false,
            })));
        };

        let configured_max = configured_max.unwrap_or_else(|| worker_capacity.current());
        validate_partitions(
            &partition_config.admission_partitions,
            &partition_config.default_admission_partition,
            configured_max,
            rc.queue_size,
        )?;

        let partition_configs: Vec<(String, super::AdmissionPartitionConfig)> = {
            let mut entries: Vec<_> = partition_config
                .admission_partitions
                .iter()
                .map(|(name, config)| (name.clone(), *config))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        };
        // Subscribe before the initial registry snapshot so worker changes
        // cannot land in the gap between startup allocation and coordinator
        // activation.
        let worker_events = registry.subscribe_events();
        let adaptive_capacity_watch = adaptive_capacity_provider
            .as_ref()
            .map(|provider| provider.subscribe_capacity_changes());
        let (initial, initial_replicas) = allocate_partition_capacities(
            &partition_configs,
            &partition_config.default_admission_partition,
            worker_capacity.current(),
            &registry,
        );
        let mut partitions = HashMap::new();
        let mut capacity_senders = Vec::with_capacity(partition_configs.len());

        for (name, config) in &partition_configs {
            let nominal_capacity = config
                .max_concurrent_requests
                .or(config.max_concurrent_requests_per_healthy_replica)
                .unwrap_or_else(|| initial.get(name).copied().unwrap_or(0));
            let partition_settings = settings.clone().for_admission_partition(
                nominal_capacity,
                configured_max,
                config.queue_size as usize,
            );
            let static_capacity = initial.get(name).copied().unwrap_or(0);
            let capacity = effective_partition_capacity(
                adaptive_capacity_provider.as_deref(),
                name,
                static_capacity,
            );
            super::metrics::set_partition_healthy_replicas(
                name,
                initial_replicas.get(name).copied().unwrap_or(0),
            );
            let scheduler = PriorityScheduler::new_with_fair_share(
                &partition_settings,
                capacity,
                fair_share.clone(),
            )
            .map_err(|e| format!("partition {name}: {e}"))?;
            let (capacity_tx, capacity_rx) = watch::channel(capacity);
            scheduler.spawn_dispatcher(capacity_rx);
            capacity_senders.push((name.clone(), capacity_tx));
            partitions.insert(
                name.clone(),
                SchedulerPartition {
                    name: Arc::from(name.as_str()),
                    scheduler,
                },
            );
            info!(
                admission.partition = %name,
                admission.max_concurrent_requests = ?config.max_concurrent_requests,
                admission.capacity_from_healthy_replicas = config.capacity_from_healthy_replicas,
                admission.healthy_replicas = initial_replicas.get(name).copied().unwrap_or(0),
                admission.initial_capacity = capacity,
                admission.queue_size = config.queue_size,
                "priority admission partition enabled"
            );
        }

        let default_partition_name = partition_config.default_admission_partition.clone();
        let default_scheduler = Arc::clone(
            &partitions
                .get(&default_partition_name)
                .ok_or_else(|| {
                    format!(
                        "default admission partition {default_partition_name:?} disappeared during startup"
                    )
                })?
                .scheduler,
        );
        spawn_partition_capacity_coordinator(PartitionCapacityCoordinator {
            capacity_watch,
            worker_capacity,
            registry,
            worker_events,
            configs: partition_configs,
            default_partition: default_partition_name.clone(),
            capacity_senders,
            adaptive_capacity_provider,
            adaptive_capacity_watch,
        });
        spawn_partition_metrics_sampler(partitions.values().cloned().collect());

        Ok(Self::Priority(Arc::new(SchedulerState {
            scheduler: default_scheduler,
            partitions,
            default_partition: Arc::from(default_partition_name),
            resolver,
            model_registry,
            rate_limiter,
            capacity_credits,
            capacity_credit_required: rc.capacity_credit_required,
        })))
    }
}

fn effective_partition_capacity(
    provider: Option<&dyn AdaptiveCapacityProvider>,
    partition: &str,
    static_capacity: u16,
) -> u16 {
    provider.map_or(static_capacity, |provider| {
        provider.effective_capacity(partition, static_capacity)
    })
}

fn spawn_partition_metrics_sampler(partitions: Vec<SchedulerPartition>) {
    let partitions: Vec<_> = partitions
        .into_iter()
        .map(|partition| (partition.name, Arc::downgrade(&partition.scheduler)))
        .collect();
    #[expect(
        clippy::disallowed_methods,
        reason = "sampler holds only weak scheduler references and exits after all partitions drop"
    )]
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(SAMPLER_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let mut live = 0_usize;
            let mut total_capacity = 0_u16;
            let mut total_queue_capacity = 0_usize;
            let mut total_inflight = [0_u16; 4];
            let mut total_queue_depth = [0_usize; 4];
            let mut total_queue_limit = [0_usize; 4];
            let mut max_retry_after = [0_u64; 4];
            let mut max_pressure = [0.0_f64; 4];

            for (name, weak) in &partitions {
                let Some(scheduler) = weak.upgrade() else {
                    continue;
                };
                live += 1;
                let snapshot = scheduler.metrics_snapshot();
                total_capacity = total_capacity.saturating_add(snapshot.capacity);
                total_queue_capacity = total_queue_capacity.saturating_add(snapshot.queue_capacity);
                let partition_inflight: u32 = snapshot
                    .inflight
                    .iter()
                    .map(|value| u32::from(*value))
                    .sum();
                let utilization = if snapshot.capacity == 0 {
                    0.0
                } else {
                    f64::from(partition_inflight) / f64::from(snapshot.capacity)
                };
                super::metrics::set_partition_capacity(name, snapshot.capacity);
                super::metrics::set_partition_utilization(name, utilization);

                for class in Class::ALL {
                    let index = class as usize;
                    super::metrics::set_partition_inflight(name, class, snapshot.inflight[index]);
                    super::metrics::set_partition_queue_depth(
                        name,
                        class,
                        snapshot.queue_depth[index],
                    );
                    super::metrics::set_partition_queue_size_limit(
                        name,
                        class,
                        snapshot.queue_limit[index],
                    );
                    total_inflight[index] =
                        total_inflight[index].saturating_add(snapshot.inflight[index]);
                    total_queue_depth[index] =
                        total_queue_depth[index].saturating_add(snapshot.queue_depth[index]);
                    total_queue_limit[index] =
                        total_queue_limit[index].saturating_add(snapshot.queue_limit[index]);
                    max_retry_after[index] =
                        max_retry_after[index].max(snapshot.retry_after_secs[index]);
                    max_pressure[index] = max_pressure[index].max(snapshot.class_pressure[index]);
                }
            }

            if live == 0 {
                break;
            }
            let aggregate_inflight: u32 =
                total_inflight.iter().map(|value| u32::from(*value)).sum();
            for class in Class::ALL {
                let index = class as usize;
                super::metrics::set_inflight(class, total_inflight[index]);
                super::metrics::set_queue_depth(class, total_queue_depth[index]);
                super::metrics::set_queue_size_limit(class, total_queue_limit[index]);
                super::metrics::set_retry_after_seconds(class, max_retry_after[index]);
                super::metrics::set_class_capacity_pressure(class, max_pressure[index]);
            }
            Metrics::set_http_admission_limit(usize::from(total_capacity));
            Metrics::set_http_admission_queue_capacity(total_queue_capacity);
            super::metrics::set_utilization(if total_capacity == 0 {
                0.0
            } else {
                f64::from(aggregate_inflight) / f64::from(total_capacity)
            });
        }
    });
}

fn validate_partitions(
    partitions: &HashMap<String, super::AdmissionPartitionConfig>,
    default_partition: &str,
    global_max: u16,
    global_queue_size: usize,
) -> Result<(), String> {
    if !partitions.contains_key(default_partition) {
        return Err(format!(
            "default admission partition {default_partition:?} is not configured"
        ));
    }
    let mut total_static_capacity = 0_u32;
    let mut total_queue = 0_u64;
    let mut dynamic_mode = None;
    for (name, config) in partitions {
        if name.is_empty() || name.trim() != name {
            return Err(format!("invalid admission partition name {name:?}"));
        }
        match (
            config.capacity_from_healthy_replicas,
            config.max_concurrent_requests,
            config.max_concurrent_requests_per_healthy_replica,
        ) {
            (true, None, None) | (true, None, Some(1..)) => {}
            (true, _, _) => {
                return Err(format!(
                    "partition {name}: replica-aware capacity must omit max_concurrent_requests; max_concurrent_requests_per_healthy_replica, when set, must be > 0"
                ));
            }
            (false, Some(limit), None) if limit > 0 => {
                total_static_capacity += u32::from(limit);
            }
            (false, _, _) => {
                return Err(format!(
                    "partition {name}: static capacity requires max_concurrent_requests > 0 and must omit max_concurrent_requests_per_healthy_replica"
                ));
            }
        }
        if let Some(expected) = dynamic_mode {
            if expected != config.capacity_from_healthy_replicas {
                return Err(
                    "admission partitions must all use static capacity or all use replica-aware capacity"
                        .to_string(),
                );
            }
        } else {
            dynamic_mode = Some(config.capacity_from_healthy_replicas);
        }
        total_queue += u64::from(config.queue_size);
    }
    if total_static_capacity > u32::from(global_max) {
        return Err(format!(
            "admission partition capacities sum to {total_static_capacity}, above global max {global_max}"
        ));
    }
    if total_queue > global_queue_size as u64 {
        return Err(format!(
            "admission partition queues sum to {total_queue}, above global queue size {global_queue_size}"
        ));
    }
    Ok(())
}

/// Allocate `target` slots in proportion to non-negative integer weights
/// using deterministic largest-remainder rounding.
fn allocate_weighted_capacities(weights: &[(String, u64)], target: u16) -> HashMap<String, u16> {
    let total_weight: u64 = weights.iter().map(|(_, weight)| *weight).sum();
    if total_weight == 0 {
        return weights.iter().map(|(name, _)| (name.clone(), 0)).collect();
    }
    let target = u64::from(target);
    let mut allocated = 0_u64;
    let mut rows: Vec<(String, u16, u64)> = weights
        .iter()
        .map(|(name, weight)| {
            let numerator = *weight * target;
            let base = (numerator / total_weight) as u16;
            allocated += u64::from(base);
            (name.clone(), base, numerator % total_weight)
        })
        .collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let mut remainder = target.saturating_sub(allocated);
    for (_, base, _) in &mut rows {
        if remainder == 0 {
            break;
        }
        *base = base.saturating_add(1);
        remainder -= 1;
    }
    rows.into_iter()
        .map(|(name, capacity, _)| (name, capacity))
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct PartitionWorkerCapacity {
    replicas: u16,
    reported_replicas: u16,
    reported_capacity: u64,
}

fn healthy_partition_worker_capacities(
    partition_names: &HashSet<String>,
    default_partition: &str,
    registry: &WorkerRegistry,
) -> HashMap<String, PartitionWorkerCapacity> {
    let mut capacities: HashMap<String, PartitionWorkerCapacity> = partition_names
        .iter()
        .map(|name| (name.clone(), PartitionWorkerCapacity::default()))
        .collect();
    for worker in registry.get_all().into_iter().filter(|w| w.is_healthy()) {
        let requested = worker
            .metadata()
            .spec
            .labels
            .get(ADMISSION_PARTITION_LABEL)
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| worker.model_id());
        let partition = if partition_names.contains(requested) {
            requested
        } else {
            default_partition
        };
        let aggregate = capacities.entry(partition.to_string()).or_default();
        aggregate.replicas = aggregate.replicas.saturating_add(1);
        if let Some(reported) = worker.max_running_requests() {
            aggregate.reported_replicas = aggregate.reported_replicas.saturating_add(1);
            aggregate.reported_capacity = aggregate
                .reported_capacity
                .saturating_add(u64::from(reported));
        }
    }
    capacities
}

/// Static mode preserves the original configured-cap behavior. Replica-aware
/// mode uses healthy replica counts as weights and distributes the entire live
/// global capacity across the currently healthy fleet.
fn allocate_partition_capacities(
    configs: &[(String, super::AdmissionPartitionConfig)],
    default_partition: &str,
    global_capacity: u16,
    registry: &WorkerRegistry,
) -> (HashMap<String, u16>, HashMap<String, u16>) {
    let replica_aware = configs
        .first()
        .is_some_and(|(_, config)| config.capacity_from_healthy_replicas);
    if !replica_aware {
        let Some(limits): Option<Vec<_>> = configs
            .iter()
            .map(|(name, config)| {
                config
                    .max_concurrent_requests
                    .map(|capacity| (name.clone(), u64::from(capacity)))
            })
            .collect()
        else {
            error!("validated static admission partition is missing its capacity");
            return (HashMap::new(), HashMap::new());
        };
        let total_limit: u64 = limits.iter().map(|(_, limit)| *limit).sum();
        let target = u64::from(global_capacity).min(total_limit) as u16;
        return (
            allocate_weighted_capacities(&limits, target),
            HashMap::new(),
        );
    }

    let names: HashSet<_> = configs.iter().map(|(name, _)| name.clone()).collect();
    let worker_capacities =
        healthy_partition_worker_capacities(&names, default_partition, registry);
    let counts: HashMap<String, u16> = worker_capacities
        .iter()
        .map(|(name, capacity)| (name.clone(), capacity.replicas))
        .collect();
    let total_replicas: u64 = worker_capacities
        .values()
        .map(|capacity| u64::from(capacity.replicas))
        .sum();
    let Some(weights): Option<Vec<_>> = configs
        .iter()
        .map(|(name, config)| {
            let capacity = worker_capacities.get(name).copied().unwrap_or_default();
            let weight =
                if let Some(per_replica) = config.max_concurrent_requests_per_healthy_replica {
                    u64::from(capacity.replicas).saturating_mul(u64::from(per_replica))
                } else if capacity.reported_replicas > 0 {
                    // Scale the observed per-replica mean across any healthy
                    // workers whose metadata discovery is still catching up.
                    capacity
                        .reported_capacity
                        .saturating_mul(u64::from(capacity.replicas))
                        / u64::from(capacity.reported_replicas)
                } else if total_replicas > 0 {
                    u64::from(global_capacity).saturating_mul(u64::from(capacity.replicas))
                        / total_replicas
                } else {
                    0
                };
            Some((name.clone(), weight))
        })
        .collect()
    else {
        error!("validated replica-aware admission partition is missing its capacity source");
        return (HashMap::new(), counts);
    };
    let desired: u64 = weights.iter().map(|(_, weight)| *weight).sum();
    let target = u64::from(global_capacity).min(desired) as u16;
    (allocate_weighted_capacities(&weights, target), counts)
}

struct PartitionCapacityCoordinator {
    capacity_watch: watch::Receiver<u16>,
    worker_capacity: Arc<WorkerCapacity>,
    registry: Arc<WorkerRegistry>,
    worker_events: broadcast::Receiver<WorkerEvent>,
    configs: Vec<(String, super::AdmissionPartitionConfig)>,
    default_partition: String,
    capacity_senders: Vec<(String, watch::Sender<u16>)>,
    adaptive_capacity_provider: Option<Arc<dyn AdaptiveCapacityProvider>>,
    adaptive_capacity_watch: Option<watch::Receiver<u64>>,
}

fn spawn_partition_capacity_coordinator(coordinator: PartitionCapacityCoordinator) {
    let PartitionCapacityCoordinator {
        mut capacity_watch,
        worker_capacity,
        registry,
        mut worker_events,
        configs,
        default_partition,
        capacity_senders,
        adaptive_capacity_provider,
        adaptive_capacity_watch,
    } = coordinator;
    #[expect(
        clippy::disallowed_methods,
        reason = "gateway-lifetime coordinator owns the capacity tracker and exits when its watch closes"
    )]
    tokio::spawn(async move {
        let _worker_capacity = worker_capacity;
        let (disabled_tx, disabled_rx) = watch::channel(0_u64);
        let _disabled_tx = disabled_tx;
        let mut adaptive_capacity_watch = adaptive_capacity_watch.unwrap_or(disabled_rx);
        loop {
            tokio::select! {
                changed = capacity_watch.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                event = worker_events.recv() => {
                    match event {
                        Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                changed = adaptive_capacity_watch.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
            }
            let (allocations, replica_counts) = allocate_partition_capacities(
                &configs,
                &default_partition,
                *capacity_watch.borrow(),
                &registry,
            );
            for (name, sender) in &capacity_senders {
                if let Some(static_capacity) = allocations.get(name) {
                    let effective_capacity = effective_partition_capacity(
                        adaptive_capacity_provider.as_deref(),
                        name,
                        *static_capacity,
                    );
                    sender.send_if_modified(|current| {
                        if *current == effective_capacity {
                            return false;
                        }
                        *current = effective_capacity;
                        true
                    });
                }
                super::metrics::set_partition_healthy_replicas(
                    name,
                    replica_counts.get(name).copied().unwrap_or(0),
                );
            }
        }
    });
}

/// Load + parse the optional priority-scheduler YAML file.
fn load_yaml(path: Option<&str>) -> Result<Option<super::PrioritySchedulerYaml>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let parsed = serde_yaml::from_str(&contents).map_err(|e| format!("parsing {path}: {e}"))?;
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Write, time::Instant};

    use axum::http::{HeaderMap, HeaderValue};
    use smg_auth::RequestId;
    use tempfile::NamedTempFile;
    use tokio::time::{sleep, Duration};

    use super::*;
    use crate::{
        middleware::scheduler::{FairShareConfig, ModelFairShareConfig, PrioritySchedulerYaml},
        worker::BasicWorkerBuilder,
    };

    #[tokio::test]
    async fn production_startup_preserves_fallback_until_credit_boundary_is_enabled() {
        let mut yaml = NamedTempFile::new().unwrap();
        write!(yaml, "not: [valid").unwrap();
        let config = RouterConfig {
            priority_scheduler_enabled: true,
            priority_scheduler_config: Some(yaml.path().to_string_lossy().into_owned()),
            ..RouterConfig::default()
        };

        let compatible = AdmissionMode::try_from_config_with_adaptive(
            &config,
            Arc::new(WorkerRegistry::new()),
            None,
            None,
        )
        .unwrap();
        assert!(matches!(compatible, AdmissionMode::Legacy));
        assert!(matches!(
            AdmissionMode::from_config(&config, Arc::new(WorkerRegistry::new()), None,),
            AdmissionMode::Legacy
        ));

        let fail_closed = RouterConfig {
            capacity_credit_generation: Some("green-1".to_string()),
            ..config
        };
        assert!(AdmissionMode::try_from_config_with_adaptive(
            &fail_closed,
            Arc::new(WorkerRegistry::new()),
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn trusted_request_model_alias_selects_the_canonical_profile() {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://alias-worker:8000")
                .model(
                    openai_protocol::model_card::ModelCard::new("deepseek-v4-flash")
                        .with_alias("deepseek-flash"),
                )
                .build(),
        );
        registry.register(worker).expect("worker should register");
        let fair_config = FairShareConfig {
            default_weight: 1.0,
            default_output_tokens: 10,
            trust_output_token_estimate_header: false,
            trust_request_model_header: true,
            tenant_weights: HashMap::new(),
            model_profiles: HashMap::from([(
                "deepseek-v4-flash".to_string(),
                ModelFairShareConfig {
                    tenant_weights: HashMap::new(),
                    other_weight: 1.0,
                },
            )]),
        };
        let yaml = PrioritySchedulerYaml {
            fair_share: Some(fair_config.clone()),
            ..Default::default()
        };
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        let state = SchedulerState {
            scheduler: PriorityScheduler::new(&settings, 1).unwrap(),
            partitions: HashMap::new(),
            default_partition: Arc::from("global"),
            resolver: Arc::new(StaticTenantPolicyResolver::from_settings(&settings)),
            model_registry: Arc::clone(&registry),
            rate_limiter: None,
            capacity_credits: None,
            capacity_credit_required: false,
        };
        let ledger = GlobalFairShare::from_config(&fair_config);
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_MODEL_HEADER,
            HeaderValue::from_static("deepseek-flash"),
        );

        assert_eq!(
            state.fair_share_profile_for(&headers, &ledger),
            FairShareProfile::Model(Arc::new("deepseek-v4-flash".to_string()))
        );
    }

    #[tokio::test]
    async fn priority_mode_applies_capacity_changes_after_startup() {
        let registry = Arc::new(WorkerRegistry::new());
        let config = RouterConfig {
            max_concurrent_requests: 256,
            priority_scheduler_enabled: true,
            ..RouterConfig::default()
        };
        let AdmissionMode::Priority(state) =
            AdmissionMode::try_build_priority(&config, Arc::clone(&registry), None, None).unwrap()
        else {
            panic!("priority scheduler should start");
        };

        let mut labels = HashMap::new();
        labels.insert("max_running_requests".to_string(), "1".to_string());
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://capacity-test:8000")
                .labels(labels)
                .status(openai_protocol::worker::WorkerStatus::Ready)
                .build(),
        );
        registry.register(worker).expect("worker should register");

        // The dispatcher must retain the tracker and apply its update. Once
        // capacity drops to one, a second System request cannot acquire a
        // slot while the first is still in flight.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut attempt = 0;
        loop {
            let first = state.scheduler.acquire_inflight(
                Class::System,
                RequestId(format!("capacity-first-{attempt}")),
            );
            let second = state.scheduler.acquire_inflight(
                Class::System,
                RequestId(format!("capacity-second-{attempt}")),
            );
            let updated = first.is_some() && second.is_none();
            drop(second);
            drop(first);

            if updated {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "scheduler did not apply the worker capacity update"
            );
            attempt += 1;
            sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn priority_mode_caps_aggregate_worker_capacity_at_configured_maximum() {
        let registry = Arc::new(WorkerRegistry::new());
        let mut labels = HashMap::new();
        labels.insert("max_running_requests".to_string(), "512".to_string());
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://capacity-test:8000")
                .labels(labels)
                .status(openai_protocol::worker::WorkerStatus::Ready)
                .build(),
        );
        registry.register(worker).expect("worker should register");

        let config = RouterConfig {
            max_concurrent_requests: 256,
            priority_scheduler_enabled: true,
            ..RouterConfig::default()
        };
        let AdmissionMode::Priority(state) =
            AdmissionMode::try_build_priority(&config, registry, None, None).unwrap()
        else {
            panic!("priority scheduler should start");
        };

        let permits: Vec<_> = (0..256)
            .map(|index| {
                state
                    .scheduler
                    .acquire_inflight(Class::System, RequestId(format!("cap-{index}")))
                    .expect("configured capacity should remain available")
            })
            .collect();
        let overflow = state
            .scheduler
            .acquire_inflight(Class::System, RequestId("cap-overflow".into()));
        assert!(overflow.is_none());
        drop(permits);
    }

    #[test]
    fn proportional_partition_allocation_preserves_global_ceiling() {
        let weights = vec![
            ("default".to_string(), 1),
            ("dsv4".to_string(), 2),
            ("kimi-k3".to_string(), 7),
        ];
        let full = allocate_weighted_capacities(&weights, 10);
        assert_eq!(full.values().copied().sum::<u16>(), 10);
        assert_eq!(full["kimi-k3"], 7);

        let drained = allocate_weighted_capacities(&weights, 5);
        assert_eq!(drained.values().copied().sum::<u16>(), 5);
        assert_eq!(drained["kimi-k3"], 3);
        assert_eq!(drained["dsv4"], 1);
        assert_eq!(drained["default"], 1);
    }

    #[test]
    fn partition_validation_rejects_budget_inflation() {
        let partitions = HashMap::from([
            (
                "default".to_string(),
                super::super::AdmissionPartitionConfig {
                    max_concurrent_requests: Some(4),
                    capacity_from_healthy_replicas: false,
                    max_concurrent_requests_per_healthy_replica: None,
                    queue_size: 2,
                },
            ),
            (
                "kimi-k3".to_string(),
                super::super::AdmissionPartitionConfig {
                    max_concurrent_requests: Some(7),
                    capacity_from_healthy_replicas: false,
                    max_concurrent_requests_per_healthy_replica: None,
                    queue_size: 3,
                },
            ),
        ]);
        assert!(validate_partitions(&partitions, "default", 10, 5)
            .unwrap_err()
            .contains("above global max"));
        assert!(validate_partitions(&partitions, "default", 11, 4)
            .unwrap_err()
            .contains("above global queue size"));
    }

    #[test]
    fn replica_aware_allocation_tracks_healthy_workers_and_private_labels() {
        let registry = WorkerRegistry::new();
        let ready = openai_protocol::worker::WorkerStatus::Ready;
        for index in 0..3 {
            let worker = Arc::new(
                BasicWorkerBuilder::new(format!("http://k3-{index}:8000"))
                    .model(openai_protocol::model_card::ModelCard::new("kimi-k3"))
                    .status(ready)
                    .build(),
            );
            registry.register(worker).unwrap();
        }
        let mut private_labels = HashMap::new();
        private_labels.insert(ADMISSION_PARTITION_LABEL.to_string(), "private".to_string());
        let private = Arc::new(
            BasicWorkerBuilder::new("http://private-k3:8000")
                .model(openai_protocol::model_card::ModelCard::new("kimi-k3"))
                .labels(private_labels)
                .status(ready)
                .build(),
        );
        registry.register(private).unwrap();
        let dsv4 = Arc::new(
            BasicWorkerBuilder::new("http://dsv4:8000")
                .model(openai_protocol::model_card::ModelCard::new("dsv4"))
                .status(ready)
                .build(),
        );
        registry.register(dsv4).unwrap();

        let dynamic = |queue_size| super::super::AdmissionPartitionConfig {
            max_concurrent_requests: None,
            capacity_from_healthy_replicas: true,
            max_concurrent_requests_per_healthy_replica: Some(100),
            queue_size,
        };
        let configs = vec![
            ("default".to_string(), dynamic(1)),
            ("dsv4".to_string(), dynamic(1)),
            ("kimi-k3".to_string(), dynamic(3)),
            ("private".to_string(), dynamic(1)),
        ];
        let (allocation, counts) =
            allocate_partition_capacities(&configs, "default", 500, &registry);
        assert_eq!(counts["kimi-k3"], 3);
        assert_eq!(counts["private"], 1);
        assert_eq!(counts["dsv4"], 1);
        assert_eq!(counts["default"], 0);
        assert_eq!(allocation["kimi-k3"], 300);
        assert_eq!(allocation["private"], 100);
        assert_eq!(allocation["dsv4"], 100);
        assert_eq!(allocation.values().copied().sum::<u16>(), 500);
    }

    #[test]
    fn replica_aware_allocation_uses_worker_reported_limits_without_override() {
        let registry = WorkerRegistry::new();
        let ready = openai_protocol::worker::WorkerStatus::Ready;
        for (url, model, limit) in [
            ("http://k3-a:8000", "kimi-k3", "64"),
            ("http://k3-b:8000", "kimi-k3", "64"),
            ("http://dsv4:8000", "dsv4", "256"),
        ] {
            let mut labels = HashMap::new();
            labels.insert("max_running_requests".to_string(), limit.to_string());
            let worker = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(openai_protocol::model_card::ModelCard::new(model))
                    .labels(labels)
                    .status(ready)
                    .build(),
            );
            registry.register(worker).unwrap();
        }

        let dynamic = |queue_size| super::super::AdmissionPartitionConfig {
            max_concurrent_requests: None,
            capacity_from_healthy_replicas: true,
            max_concurrent_requests_per_healthy_replica: None,
            queue_size,
        };
        let configs = vec![
            ("default".to_string(), dynamic(1)),
            ("dsv4".to_string(), dynamic(1)),
            ("kimi-k3".to_string(), dynamic(1)),
        ];
        let (allocation, counts) =
            allocate_partition_capacities(&configs, "default", 384, &registry);
        assert_eq!(counts["kimi-k3"], 2);
        assert_eq!(counts["dsv4"], 1);
        assert_eq!(allocation["kimi-k3"], 128);
        assert_eq!(allocation["dsv4"], 256);
        assert_eq!(allocation["default"], 0);
    }

    #[test]
    fn replica_aware_allocation_sends_unconfigured_models_to_default() {
        let registry = WorkerRegistry::new();
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://new-model:8000")
                .model(openai_protocol::model_card::ModelCard::new("new-model"))
                .status(openai_protocol::worker::WorkerStatus::Ready)
                .build(),
        );
        registry.register(worker).unwrap();
        let dynamic = super::super::AdmissionPartitionConfig {
            max_concurrent_requests: None,
            capacity_from_healthy_replicas: true,
            max_concurrent_requests_per_healthy_replica: Some(64),
            queue_size: 1,
        };
        let configs = vec![
            ("default".to_string(), dynamic),
            ("kimi-k3".to_string(), dynamic),
        ];
        let (allocation, counts) =
            allocate_partition_capacities(&configs, "default", 64, &registry);
        assert_eq!(counts["default"], 1);
        assert_eq!(allocation["default"], 64);
        assert_eq!(allocation["kimi-k3"], 0);
    }

    #[tokio::test]
    async fn replica_aware_coordinator_reacts_to_worker_registration() {
        let mut yaml = NamedTempFile::new().unwrap();
        write!(
            yaml,
            r#"
admission_partitions:
  kimi-k3:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 3
    queue_size: 3
  dsv4:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 3
    queue_size: 1
  private:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 3
    queue_size: 0
  default:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 3
    queue_size: 0
default_admission_partition: default
"#
        )
        .unwrap();
        let registry = Arc::new(WorkerRegistry::new());
        let config = RouterConfig {
            max_concurrent_requests: 9,
            queue_size: 4,
            priority_scheduler_enabled: true,
            priority_scheduler_config: Some(yaml.path().to_string_lossy().into_owned()),
            ..RouterConfig::default()
        };
        let AdmissionMode::Priority(state) =
            AdmissionMode::try_build_priority(&config, Arc::clone(&registry), None, None).unwrap()
        else {
            panic!("priority scheduler should start");
        };

        let mut k3_headers = HeaderMap::new();
        k3_headers.insert(
            ADMISSION_PARTITION_HEADER,
            HeaderValue::from_static("kimi-k3"),
        );
        assert!(state
            .partition_for(&k3_headers)
            .scheduler
            .acquire_inflight(Class::System, RequestId("empty-fleet".into()))
            .is_none());

        let mut first_k3 = None;
        for index in 0..3 {
            let worker_id = registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("http://k3-live-{index}:8000"))
                        .model(openai_protocol::model_card::ModelCard::new("kimi-k3"))
                        .status(openai_protocol::worker::WorkerStatus::Ready)
                        .build(),
                ))
                .unwrap();
            if index == 0 {
                first_k3 = Some(worker_id);
            }
        }
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("http://dsv4-live:8000")
                    .model(openai_protocol::model_card::ModelCard::new("dsv4"))
                    .status(openai_protocol::worker::WorkerStatus::Ready)
                    .build(),
            ))
            .unwrap();

        let k3 = state.partition_for(&k3_headers);
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let permits: Vec<_> = (0..7)
                .filter_map(|index| {
                    k3.scheduler
                        .acquire_inflight(Class::System, RequestId(format!("dynamic-k3-{index}")))
                })
                .collect();
            let full = permits.len() == 7
                && k3
                    .scheduler
                    .acquire_inflight(Class::System, RequestId("dynamic-k3-full".into()))
                    .is_none();
            drop(permits);
            if full {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "partition coordinator did not apply replica weights"
            );
            sleep(Duration::from_millis(10)).await;
        }

        let mut private_labels = HashMap::new();
        private_labels.insert(ADMISSION_PARTITION_LABEL.to_string(), "private".to_string());
        let moved = Arc::new(
            BasicWorkerBuilder::new("http://k3-live-0:8000")
                .model(openai_protocol::model_card::ModelCard::new("kimi-k3"))
                .labels(private_labels)
                .status(openai_protocol::worker::WorkerStatus::Ready)
                .build(),
        );
        assert!(registry.replace(&first_k3.unwrap(), moved));

        let mut private_headers = HeaderMap::new();
        private_headers.insert(
            ADMISSION_PARTITION_HEADER,
            HeaderValue::from_static("private"),
        );
        let private = state.partition_for(&private_headers);
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let first = private
                .scheduler
                .acquire_inflight(Class::System, RequestId("private-first".into()));
            let second = private
                .scheduler
                .acquire_inflight(Class::System, RequestId("private-second".into()));
            let overflow = private
                .scheduler
                .acquire_inflight(Class::System, RequestId("private-full".into()));
            let moved = first.is_some() && second.is_some() && overflow.is_none();
            drop(overflow);
            drop(second);
            drop(first);
            if moved {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "partition coordinator did not apply worker label replacement"
            );
            sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn saturated_model_partition_does_not_block_another_model() {
        let mut yaml = NamedTempFile::new().unwrap();
        write!(
            yaml,
            r#"
admission_partitions:
  kimi-k3:
    max_concurrent_requests: 6
    queue_size: 3
  deepseek-v4-flash-0731:
    max_concurrent_requests: 1
    queue_size: 1
  private:
    max_concurrent_requests: 1
    queue_size: 0
  default:
    max_concurrent_requests: 1
    queue_size: 0
default_admission_partition: default
"#
        )
        .unwrap();
        let config = RouterConfig {
            max_concurrent_requests: 9,
            queue_size: 4,
            priority_scheduler_enabled: true,
            priority_scheduler_config: Some(yaml.path().to_string_lossy().into_owned()),
            ..RouterConfig::default()
        };
        let AdmissionMode::Priority(state) =
            AdmissionMode::try_build_priority(&config, Arc::new(WorkerRegistry::new()), None, None)
                .unwrap()
        else {
            panic!("priority scheduler should start");
        };

        let mut k3_headers = HeaderMap::new();
        k3_headers.insert(
            ADMISSION_PARTITION_HEADER,
            HeaderValue::from_static("kimi-k3"),
        );
        let k3 = state.partition_for(&k3_headers);
        let k3_permits: Vec<_> = (0..6)
            .map(|index| {
                k3.scheduler
                    .acquire_inflight(Class::System, RequestId(format!("k3-{index}")))
                    .expect("K3 partition slot")
            })
            .collect();
        assert!(k3
            .scheduler
            .acquire_inflight(Class::System, RequestId("k3-full".into()))
            .is_none());

        let mut dsv4_headers = HeaderMap::new();
        dsv4_headers.insert(
            ADMISSION_PARTITION_HEADER,
            HeaderValue::from_static("deepseek-v4-flash-0731"),
        );
        let dsv4 = state.partition_for(&dsv4_headers);
        assert_eq!(&*dsv4.name, "deepseek-v4-flash-0731");
        let dsv4_permit = dsv4
            .scheduler
            .acquire_inflight(Class::System, RequestId("dsv4".into()));
        assert!(dsv4_permit.is_some());

        let mut unknown_headers = HeaderMap::new();
        unknown_headers.insert(
            ADMISSION_PARTITION_HEADER,
            HeaderValue::from_static("new-model"),
        );
        assert_eq!(&*state.partition_for(&unknown_headers).name, "default");
        drop(dsv4_permit);
        drop(k3_permits);
    }
}
