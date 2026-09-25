//! Process-wide output-token accounting and weighted service with local queues.
//!
//! Every priority-scheduler partition receives the same [`Arc<GlobalFairShare>`].
//! Lifetime charged-token accounting spans that process. Flat configuration
//! also shares canonical tenant virtual finish across partitions. Optional
//! model profiles instead keep virtual service and active-set time model-local,
//! with a hierarchical outer named-tenant/other-bucket policy. Each partition
//! stores only local queued/reservation membership. Queued work is registered
//! before a partition asks for its next local waiter. This keeps every model
//! pool work-conserving:
//! unrelated or non-fungible capacity is never idled to repay another tenant's
//! debt.
//!
//! The ledger is process-local. It spans all model/admission partitions and
//! all workload types that resolve to the same tenant key in one SMG process;
//! it does not coordinate separate gateways or overlapping blue/green
//! processes. Because model pools are non-fungible, it also cannot guarantee
//! exact aggregate percentages when users target disjoint pools. Flat service
//! debt can follow a tenant across pools; model-profile debt never crosses
//! models. Neither mode can force a disjoint pool to idle.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use parking_lot::Mutex;

use super::{Class, FairShareConfig, SchedulerSettings};
use crate::tenant::TenantKey;

/// Trusted estimate injected by the authenticated Comet proxy.
pub const OUTPUT_TOKEN_ESTIMATE_HEADER: &str = "x-smg-output-token-estimate";

/// Trusted canonical-model selector injected by an authenticated proxy.
pub const REQUEST_MODEL_HEADER: &str = "x-smg-request-model";

/// Scheduling-debt scope selected for one request.
///
/// `Global` preserves the original process-wide flat ledger. A configured
/// model profile has its own virtual clock and hierarchical service state,
/// while charged/reserved token accounting remains process-global.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum FairShareProfile {
    Global,
    Model(Arc<String>),
}

impl FairShareProfile {
    #[must_use]
    pub(crate) fn metric_label(&self) -> &str {
        match self {
            Self::Global => "global",
            Self::Model(model) => model,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementKind {
    Observed,
    MissingUsage,
    Interrupted,
}

#[derive(Debug, Default)]
struct TenantAccounting {
    charged_output_tokens: u64,
    reserved_output_tokens: u64,
}

#[derive(Debug, Default)]
struct TenantService {
    active_reservations: u64,
    queued: usize,
    virtual_finish: f64,
}

#[derive(Debug, Default)]
struct ScopeTenant {
    active_reservations: u64,
    queued: [usize; 4],
}

#[derive(Debug, Default)]
struct ScopeLedger {
    tenants: HashMap<TenantKey, ScopeTenant>,
    model_tenants: HashMap<Arc<String>, HashMap<TenantKey, ScopeTenant>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ModelBucket {
    Tenant(TenantKey),
    Other,
}

#[derive(Debug)]
struct ServiceClock<K> {
    entries: HashMap<K, TenantService>,
    system_virtual_time: f64,
}

impl<K> Default for ServiceClock<K> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            system_virtual_time: 0.0,
        }
    }
}

#[derive(Debug, Default)]
struct ModelServiceLedger {
    outer: ServiceClock<ModelBucket>,
    other: ServiceClock<TenantKey>,
}

#[derive(Debug, Default)]
struct LedgerState {
    accounting: HashMap<TenantKey, TenantAccounting>,
    service: HashMap<TenantKey, TenantService>,
    scopes: HashMap<u64, ScopeLedger>,
    system_virtual_time: f64,
    model_service: HashMap<Arc<String>, ModelServiceLedger>,
}

/// One local candidate offered by a partition queue.
pub(crate) struct FairShareCandidate<'a> {
    pub index: usize,
    pub tenant: &'a TenantKey,
    pub estimated_output_tokens: u32,
}

#[derive(Debug)]
struct ModelFairSharePolicy {
    canonical_model: Arc<String>,
    tenant_weights: HashMap<TenantKey, f64>,
    other_weight: f64,
}

/// The candidate selected by the shared ledger, plus its provisional charge.
pub(crate) struct FairShareSelection {
    pub index: usize,
    pub reservation: FairShareReservation,
}

/// Shared weighted-service ledger for every scheduler partition in one router.
pub struct GlobalFairShare {
    default_weight: f64,
    default_output_tokens: u32,
    trust_output_token_estimate_header: bool,
    trust_request_model_header: bool,
    tenant_weights: HashMap<TenantKey, f64>,
    model_profiles: HashMap<String, ModelFairSharePolicy>,
    metric_tenants: HashSet<TenantKey>,
    state: Mutex<LedgerState>,
    next_scope: AtomicU64,
}

impl std::fmt::Debug for GlobalFairShare {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalFairShare")
            .field("default_weight", &self.default_weight)
            .field("default_output_tokens", &self.default_output_tokens)
            .field(
                "trust_output_token_estimate_header",
                &self.trust_output_token_estimate_header,
            )
            .field(
                "trust_request_model_header",
                &self.trust_request_model_header,
            )
            .field("tenant_weights", &self.tenant_weights)
            .field("model_profiles", &self.model_profiles)
            .finish_non_exhaustive()
    }
}

impl GlobalFairShare {
    #[must_use]
    pub fn from_settings(settings: &SchedulerSettings) -> Option<Self> {
        settings.fair_share_config().map(|config| {
            let mut ledger = Self::from_config(config);
            let mut metric_tenants: Vec<_> = ledger
                .tenant_weights
                .keys()
                .chain(
                    ledger
                        .model_profiles
                        .values()
                        .flat_map(|profile| profile.tenant_weights.keys()),
                )
                .cloned()
                .collect();
            metric_tenants.sort_by(|left, right| left.as_str().cmp(right.as_str()));
            metric_tenants.dedup();
            metric_tenants.truncate(settings.tenant_metric_top_n as usize);
            ledger.metric_tenants = metric_tenants.into_iter().collect();
            ledger
        })
    }

    #[must_use]
    pub fn from_config(config: &FairShareConfig) -> Self {
        let tenant_weights: HashMap<_, _> = config
            .tenant_weights
            .iter()
            .map(|(tenant, weight)| (TenantKey::new(tenant), *weight))
            .collect();
        let model_profiles: HashMap<String, ModelFairSharePolicy> = config
            .model_profiles
            .iter()
            .map(|(model, profile)| {
                let tenant_weights = profile
                    .tenant_weights
                    .iter()
                    .map(|(tenant, weight)| (TenantKey::new(tenant), *weight))
                    .collect();
                let canonical_model = Arc::new(model.clone());
                (
                    model.clone(),
                    ModelFairSharePolicy {
                        canonical_model,
                        tenant_weights,
                        other_weight: profile.other_weight,
                    },
                )
            })
            .collect();
        let metric_tenants = tenant_weights
            .keys()
            .chain(
                model_profiles
                    .values()
                    .flat_map(|profile: &ModelFairSharePolicy| profile.tenant_weights.keys()),
            )
            .cloned()
            .collect();
        Self {
            default_weight: config.default_weight,
            default_output_tokens: config.default_output_tokens,
            trust_output_token_estimate_header: config.trust_output_token_estimate_header,
            trust_request_model_header: config.trust_request_model_header,
            tenant_weights,
            model_profiles,
            metric_tenants,
            state: Mutex::new(LedgerState::default()),
            next_scope: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn default_output_tokens(&self) -> u32 {
        self.default_output_tokens
    }

    #[must_use]
    pub fn trusts_output_token_estimate_header(&self) -> bool {
        self.trust_output_token_estimate_header
    }

    #[must_use]
    pub fn trusts_request_model_header(&self) -> bool {
        self.trust_request_model_header
    }

    #[must_use]
    pub fn has_model_profiles(&self) -> bool {
        !self.model_profiles.is_empty()
    }

    #[must_use]
    pub(crate) fn profile_for_model(&self, model: &str) -> FairShareProfile {
        self.model_profiles
            .get(model)
            .map(|policy| FairShareProfile::Model(Arc::clone(&policy.canonical_model)))
            .unwrap_or(FairShareProfile::Global)
    }

    pub(crate) fn new_scope(&self) -> u64 {
        self.next_scope.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn record_queue_wait(
        &self,
        profile: &FairShareProfile,
        tenant: &TenantKey,
        class: Class,
        wait: Duration,
    ) {
        super::metrics::record_fair_share_queue_wait(
            profile.metric_label(),
            self.metric_tenant(tenant),
            class,
            wait,
        );
    }

    #[cfg(test)]
    pub(crate) fn register_waiter(&self, scope_id: u64, tenant: &TenantKey, class: Class) {
        self.register_waiter_in_profile(scope_id, &FairShareProfile::Global, tenant, class);
    }

    pub(crate) fn register_waiter_in_profile(
        &self,
        scope_id: u64,
        profile: &FairShareProfile,
        tenant: &TenantKey,
        class: Class,
    ) {
        let FairShareProfile::Model(model) = profile else {
            self.register_global_waiter(scope_id, tenant, class);
            return;
        };
        let Some(policy) = self.model_profiles.get(model.as_str()) else {
            self.register_global_waiter(scope_id, tenant, class);
            return;
        };
        let bucket = if policy.tenant_weights.contains_key(tenant) {
            ModelBucket::Tenant(tenant.clone())
        } else {
            ModelBucket::Other
        };
        let is_other = bucket == ModelBucket::Other;
        let mut state = self.state.lock();
        let ledger = state.model_service.entry(Arc::clone(model)).or_default();
        Self::register_service(&mut ledger.outer, bucket);
        if is_other {
            Self::register_service(&mut ledger.other, tenant.clone());
        }
        state
            .scopes
            .entry(scope_id)
            .or_default()
            .model_tenants
            .entry(Arc::clone(model))
            .or_default()
            .entry(tenant.clone())
            .or_default()
            .queued[class as usize] += 1;
        drop(state);
        if is_other {
            super::metrics::record_fair_share_unknown_tenant(self.metric_tenant(tenant));
        }
    }

    fn register_global_waiter(&self, scope_id: u64, tenant: &TenantKey, class: Class) {
        let unknown = !self.tenant_weights.contains_key(tenant);
        let mut state = self.state.lock();
        let system_virtual_time = state.system_virtual_time;
        let service = state.service.entry(tenant.clone()).or_default();
        if !Self::is_active(service) {
            // Start-time fair queueing: idle tenants join the current epoch.
            // They neither retain stale low credit nor inherit solo service as
            // debt against users that were not backlogged at the time.
            service.virtual_finish = service.virtual_finish.max(system_virtual_time);
        }
        service.queued = service.queued.saturating_add(1);
        state
            .scopes
            .entry(scope_id)
            .or_default()
            .tenants
            .entry(tenant.clone())
            .or_default()
            .queued[class as usize] += 1;
        Self::advance_system_virtual_time(&mut state);
        drop(state);
        if unknown {
            super::metrics::record_fair_share_unknown_tenant(self.metric_tenant(tenant));
        }
    }

    pub(crate) fn remove_waiter_in_profile(
        &self,
        scope_id: u64,
        profile: &FairShareProfile,
        tenant: &TenantKey,
        class: Class,
    ) {
        let FairShareProfile::Model(model) = profile else {
            self.remove_global_waiter(scope_id, tenant, class);
            return;
        };
        let Some(policy) = self.model_profiles.get(model.as_str()) else {
            self.remove_global_waiter(scope_id, tenant, class);
            return;
        };
        let bucket = if policy.tenant_weights.contains_key(tenant) {
            ModelBucket::Tenant(tenant.clone())
        } else {
            ModelBucket::Other
        };
        let is_other = bucket == ModelBucket::Other;
        let mut state = self.state.lock();
        let Some(membership) = state
            .scopes
            .get_mut(&scope_id)
            .and_then(|scope| scope.model_tenants.get_mut(model))
            .and_then(|tenants| tenants.get_mut(tenant))
        else {
            return;
        };
        let queued = &mut membership.queued[class as usize];
        debug_assert!(*queued > 0, "model fair-share waiter removed below zero");
        if *queued == 0 {
            return;
        }
        *queued -= 1;
        let Some(ledger) = state.model_service.get_mut(model) else {
            return;
        };
        Self::remove_service_waiter(&mut ledger.outer, &bucket);
        if is_other {
            Self::remove_service_waiter(&mut ledger.other, tenant);
        }
    }

    fn remove_global_waiter(&self, scope_id: u64, tenant: &TenantKey, class: Class) {
        let mut state = self.state.lock();
        {
            let Some(scope) = state.scopes.get_mut(&scope_id) else {
                return;
            };
            let Some(membership) = scope.tenants.get_mut(tenant) else {
                return;
            };
            let queued = &mut membership.queued[class as usize];
            debug_assert!(*queued > 0, "fair-share waiter removed below zero");
            if *queued == 0 {
                return;
            }
            *queued -= 1;
        }
        if let Some(service) = state.service.get_mut(tenant) {
            service.queued = service.queued.saturating_sub(1);
        }
        Self::advance_system_virtual_time(&mut state);
        drop(state);
    }

    /// Reserve the least-served candidate eligible for this local partition.
    ///
    /// Selection never consults tenants that are queued only in another
    /// partition. That is the work-conserving boundary for non-fungible model
    /// pools: a local slot is never idled for work that cannot use it.
    #[cfg(test)]
    pub(crate) fn reserve_local_candidate(
        self: &Arc<Self>,
        scope_id: u64,
        class: Class,
        candidates: &[FairShareCandidate<'_>],
    ) -> Option<FairShareSelection> {
        self.reserve_local_candidate_in_profile(
            scope_id,
            &FairShareProfile::Global,
            class,
            candidates,
        )
    }

    pub(crate) fn reserve_local_candidate_in_profile(
        self: &Arc<Self>,
        scope_id: u64,
        profile: &FairShareProfile,
        class: Class,
        candidates: &[FairShareCandidate<'_>],
    ) -> Option<FairShareSelection> {
        match profile {
            FairShareProfile::Global => self.reserve_global_candidate(scope_id, class, candidates),
            FairShareProfile::Model(model) if self.model_profiles.contains_key(model.as_str()) => {
                self.reserve_model_candidate(scope_id, model, class, candidates)
            }
            FairShareProfile::Model(_) => {
                self.reserve_global_candidate(scope_id, class, candidates)
            }
        }
    }

    fn reserve_global_candidate(
        self: &Arc<Self>,
        scope_id: u64,
        class: Class,
        candidates: &[FairShareCandidate<'_>],
    ) -> Option<FairShareSelection> {
        let mut state = self.state.lock();
        let class_index = class as usize;
        let selected = {
            let scope = state.scopes.get(&scope_id)?;
            candidates
                .iter()
                .filter_map(|candidate| {
                    let membership = scope.tenants.get(candidate.tenant)?;
                    if membership.queued[class_index] == 0 {
                        return None;
                    }
                    let service = state.service.get(candidate.tenant)?;
                    Some((candidate, service.virtual_finish))
                })
                .min_by(|(left, left_service), (right, right_service)| {
                    left_service
                        .total_cmp(right_service)
                        .then_with(|| left.tenant.as_str().cmp(right.tenant.as_str()))
                        .then_with(|| left.index.cmp(&right.index))
                })?
                .0
        };

        let tenant = selected.tenant.clone();
        let estimated_output_tokens = selected.estimated_output_tokens.max(1);
        let membership = state.scopes.get_mut(&scope_id)?.tenants.get_mut(&tenant)?;
        debug_assert!(membership.queued[class_index] > 0);
        membership.queued[class_index] = membership.queued[class_index].saturating_sub(1);
        membership.active_reservations = membership.active_reservations.saturating_add(1);
        let service = state.service.get_mut(&tenant)?;
        service.queued = service.queued.saturating_sub(1);
        service.active_reservations = service.active_reservations.saturating_add(1);
        service.virtual_finish += f64::from(estimated_output_tokens) / self.weight(&tenant);
        let virtual_service = service.virtual_finish;
        Self::advance_system_virtual_time(&mut state);
        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_add(u64::from(estimated_output_tokens));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);

        super::metrics::set_fair_share_virtual_finish(
            "global",
            self.metric_tenant(&tenant),
            virtual_service,
        );
        super::metrics::set_fair_share_reserved_output_tokens(
            self.metric_tenant(&tenant),
            reserved_output_tokens,
        );

        Some(FairShareSelection {
            index: selected.index,
            reservation: FairShareReservation {
                ledger: Arc::clone(self),
                tenant,
                scope_id,
                profile: FairShareProfile::Global,
                estimated_output_tokens,
                settled: false,
            },
        })
    }

    fn reserve_model_candidate(
        self: &Arc<Self>,
        scope_id: u64,
        model: &Arc<String>,
        class: Class,
        candidates: &[FairShareCandidate<'_>],
    ) -> Option<FairShareSelection> {
        let policy = self.model_profiles.get(model.as_str())?;
        let class_index = class as usize;
        let mut state = self.state.lock();
        let selected_bucket = {
            let scope_tenants = state.scopes.get(&scope_id)?.model_tenants.get(model)?;
            let ledger = state.model_service.get(model)?;
            candidates
                .iter()
                .filter_map(|candidate| {
                    let membership = scope_tenants.get(candidate.tenant)?;
                    if membership.queued[class_index] == 0 {
                        return None;
                    }
                    let bucket = Self::model_bucket(policy, candidate.tenant);
                    let service = ledger.outer.entries.get(&bucket)?;
                    Some((bucket, service.virtual_finish))
                })
                .min_by(|(left_bucket, left_finish), (right_bucket, right_finish)| {
                    left_finish
                        .total_cmp(right_finish)
                        .then_with(|| Self::compare_model_buckets(left_bucket, right_bucket))
                })?
                .0
        };

        let selected = {
            let scope_tenants = state.scopes.get(&scope_id)?.model_tenants.get(model)?;
            let ledger = state.model_service.get(model)?;
            candidates
                .iter()
                .filter(|candidate| {
                    scope_tenants
                        .get(candidate.tenant)
                        .is_some_and(|membership| membership.queued[class_index] > 0)
                        && Self::model_bucket(policy, candidate.tenant) == selected_bucket
                })
                .min_by(|left, right| {
                    let left_finish = if selected_bucket == ModelBucket::Other {
                        ledger
                            .other
                            .entries
                            .get(left.tenant)
                            .map_or(f64::INFINITY, |service| service.virtual_finish)
                    } else {
                        0.0
                    };
                    let right_finish = if selected_bucket == ModelBucket::Other {
                        ledger
                            .other
                            .entries
                            .get(right.tenant)
                            .map_or(f64::INFINITY, |service| service.virtual_finish)
                    } else {
                        0.0
                    };
                    left_finish
                        .total_cmp(&right_finish)
                        .then_with(|| left.tenant.as_str().cmp(right.tenant.as_str()))
                        .then_with(|| left.index.cmp(&right.index))
                })?
        };

        let tenant = selected.tenant.clone();
        let selected_index = selected.index;
        let estimated_output_tokens = selected.estimated_output_tokens.max(1);
        let membership = state
            .scopes
            .get_mut(&scope_id)?
            .model_tenants
            .get_mut(model)?
            .get_mut(&tenant)?;
        debug_assert!(membership.queued[class_index] > 0);
        membership.queued[class_index] = membership.queued[class_index].saturating_sub(1);
        membership.active_reservations = membership.active_reservations.saturating_add(1);

        let ledger = state.model_service.get_mut(model)?;
        let outer_weight = self.model_bucket_weight(policy, &selected_bucket);
        let outer_service = ledger.outer.entries.get_mut(&selected_bucket)?;
        outer_service.queued = outer_service.queued.saturating_sub(1);
        outer_service.active_reservations = outer_service.active_reservations.saturating_add(1);
        outer_service.virtual_finish += f64::from(estimated_output_tokens) / outer_weight;
        let outer_virtual_finish = outer_service.virtual_finish;

        let inner_virtual_finish = if selected_bucket == ModelBucket::Other {
            let service = ledger.other.entries.get_mut(&tenant)?;
            service.queued = service.queued.saturating_sub(1);
            service.active_reservations = service.active_reservations.saturating_add(1);
            service.virtual_finish += f64::from(estimated_output_tokens) / self.default_weight;
            Some(service.virtual_finish)
        } else {
            None
        };
        Self::advance_service_clock(&mut ledger.outer);
        if selected_bucket == ModelBucket::Other {
            Self::advance_service_clock(&mut ledger.other);
        }

        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_add(u64::from(estimated_output_tokens));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);

        super::metrics::set_fair_share_virtual_finish(
            model,
            self.metric_tenant(&tenant),
            inner_virtual_finish.unwrap_or(outer_virtual_finish),
        );
        if selected_bucket == ModelBucket::Other {
            super::metrics::set_fair_share_other_bucket_virtual_finish(model, outer_virtual_finish);
        }
        super::metrics::set_fair_share_reserved_output_tokens(
            self.metric_tenant(&tenant),
            reserved_output_tokens,
        );

        Some(FairShareSelection {
            index: selected_index,
            reservation: FairShareReservation {
                ledger: Arc::clone(self),
                tenant,
                scope_id,
                profile: FairShareProfile::Model(Arc::clone(model)),
                estimated_output_tokens,
                settled: false,
            },
        })
    }

    fn weight(&self, tenant: &TenantKey) -> f64 {
        self.tenant_weights
            .get(tenant)
            .copied()
            .unwrap_or(self.default_weight)
    }

    fn model_bucket(policy: &ModelFairSharePolicy, tenant: &TenantKey) -> ModelBucket {
        if policy.tenant_weights.contains_key(tenant) {
            ModelBucket::Tenant(tenant.clone())
        } else {
            ModelBucket::Other
        }
    }

    fn model_bucket_weight(&self, policy: &ModelFairSharePolicy, bucket: &ModelBucket) -> f64 {
        match bucket {
            ModelBucket::Tenant(tenant) => policy
                .tenant_weights
                .get(tenant)
                .copied()
                .unwrap_or(self.default_weight),
            ModelBucket::Other => policy.other_weight,
        }
    }

    fn compare_model_buckets(left: &ModelBucket, right: &ModelBucket) -> std::cmp::Ordering {
        match (left, right) {
            (ModelBucket::Tenant(left), ModelBucket::Tenant(right)) => {
                left.as_str().cmp(right.as_str())
            }
            (ModelBucket::Tenant(_), ModelBucket::Other) => std::cmp::Ordering::Less,
            (ModelBucket::Other, ModelBucket::Tenant(_)) => std::cmp::Ordering::Greater,
            (ModelBucket::Other, ModelBucket::Other) => std::cmp::Ordering::Equal,
        }
    }

    fn is_active(service: &TenantService) -> bool {
        service.active_reservations > 0 || service.queued > 0
    }

    fn register_service<K>(clock: &mut ServiceClock<K>, key: K)
    where
        K: Eq + std::hash::Hash,
    {
        let system_virtual_time = clock.system_virtual_time;
        let service = clock.entries.entry(key).or_default();
        if !Self::is_active(service) {
            service.virtual_finish = service.virtual_finish.max(system_virtual_time);
        }
        service.queued = service.queued.saturating_add(1);
        Self::advance_service_clock(clock);
    }

    fn remove_service_waiter<K>(clock: &mut ServiceClock<K>, key: &K)
    where
        K: Eq + std::hash::Hash,
    {
        if let Some(service) = clock.entries.get_mut(key) {
            service.queued = service.queued.saturating_sub(1);
        }
        Self::advance_service_clock(clock);
    }

    fn advance_service_clock<K>(clock: &mut ServiceClock<K>) {
        if let Some(active_minimum) = clock
            .entries
            .values()
            .filter(|service| Self::is_active(service))
            .map(|service| service.virtual_finish)
            .min_by(f64::total_cmp)
        {
            clock.system_virtual_time = clock.system_virtual_time.max(active_minimum);
        }
    }

    fn advance_system_virtual_time(state: &mut LedgerState) {
        if let Some(active_minimum) = state
            .service
            .values()
            .filter(|service| Self::is_active(service))
            .map(|service| service.virtual_finish)
            .min_by(f64::total_cmp)
        {
            state.system_virtual_time = state.system_virtual_time.max(active_minimum);
        }
    }

    fn metric_tenant<'a>(&'a self, tenant: &'a TenantKey) -> &'a str {
        if self.metric_tenants.contains(tenant) {
            tenant.as_str()
        } else {
            "other"
        }
    }

    fn cancel_reservation(
        &self,
        scope_id: u64,
        profile: &FairShareProfile,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
    ) {
        match profile {
            FairShareProfile::Global => {
                self.cancel_global_reservation(scope_id, tenant, estimated_output_tokens);
            }
            FairShareProfile::Model(model) if self.model_profiles.contains_key(model.as_str()) => {
                self.cancel_model_reservation(scope_id, model, tenant, estimated_output_tokens);
            }
            FairShareProfile::Model(_) => {
                self.cancel_global_reservation(scope_id, tenant, estimated_output_tokens);
            }
        }
    }

    fn cancel_global_reservation(
        &self,
        scope_id: u64,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
    ) {
        let mut state = self.state.lock();
        {
            let Some(membership) = state
                .scopes
                .get_mut(&scope_id)
                .and_then(|scope| scope.tenants.get_mut(tenant))
            else {
                return;
            };
            if membership.active_reservations == 0 {
                return;
            }
            membership.active_reservations -= 1;
        }
        let system_virtual_time = state.system_virtual_time;
        let Some(service) = state.service.get_mut(tenant) else {
            return;
        };
        service.active_reservations = service.active_reservations.saturating_sub(1);
        service.virtual_finish = (service.virtual_finish
            - f64::from(estimated_output_tokens) / self.weight(tenant))
        .max(system_virtual_time);
        let virtual_service = service.virtual_finish;
        Self::advance_system_virtual_time(&mut state);
        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_sub(u64::from(estimated_output_tokens));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);
        super::metrics::set_fair_share_virtual_finish(
            "global",
            self.metric_tenant(tenant),
            virtual_service,
        );
        super::metrics::set_fair_share_reserved_output_tokens(
            self.metric_tenant(tenant),
            reserved_output_tokens,
        );
    }

    fn cancel_model_reservation(
        &self,
        scope_id: u64,
        model: &Arc<String>,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
    ) {
        let Some(policy) = self.model_profiles.get(model.as_str()) else {
            return;
        };
        let bucket = Self::model_bucket(policy, tenant);
        let is_other = bucket == ModelBucket::Other;
        let mut state = self.state.lock();
        let Some(membership) = state
            .scopes
            .get_mut(&scope_id)
            .and_then(|scope| scope.model_tenants.get_mut(model))
            .and_then(|tenants| tenants.get_mut(tenant))
        else {
            return;
        };
        if membership.active_reservations == 0 {
            return;
        }
        membership.active_reservations -= 1;

        let Some(ledger) = state.model_service.get_mut(model) else {
            return;
        };
        let outer_weight = self.model_bucket_weight(policy, &bucket);
        let outer_system_virtual_time = ledger.outer.system_virtual_time;
        let Some(outer_service) = ledger.outer.entries.get_mut(&bucket) else {
            return;
        };
        outer_service.active_reservations = outer_service.active_reservations.saturating_sub(1);
        outer_service.virtual_finish = (outer_service.virtual_finish
            - f64::from(estimated_output_tokens) / outer_weight)
            .max(outer_system_virtual_time);
        let outer_virtual_finish = outer_service.virtual_finish;
        let inner_virtual_finish = if is_other {
            let inner_system_virtual_time = ledger.other.system_virtual_time;
            let Some(service) = ledger.other.entries.get_mut(tenant) else {
                return;
            };
            service.active_reservations = service.active_reservations.saturating_sub(1);
            service.virtual_finish = (service.virtual_finish
                - f64::from(estimated_output_tokens) / self.default_weight)
                .max(inner_system_virtual_time);
            Some(service.virtual_finish)
        } else {
            None
        };
        Self::advance_service_clock(&mut ledger.outer);
        if is_other {
            Self::advance_service_clock(&mut ledger.other);
        }
        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_sub(u64::from(estimated_output_tokens));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);

        super::metrics::set_fair_share_virtual_finish(
            model,
            self.metric_tenant(tenant),
            inner_virtual_finish.unwrap_or(outer_virtual_finish),
        );
        if is_other {
            super::metrics::set_fair_share_other_bucket_virtual_finish(model, outer_virtual_finish);
        }
        super::metrics::set_fair_share_reserved_output_tokens(
            self.metric_tenant(tenant),
            reserved_output_tokens,
        );
    }

    fn settle_reservation(
        &self,
        scope_id: u64,
        profile: &FairShareProfile,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
        observed_output_tokens: Option<u32>,
        kind: SettlementKind,
    ) {
        match profile {
            FairShareProfile::Global => self.settle_global_reservation(
                scope_id,
                tenant,
                estimated_output_tokens,
                observed_output_tokens,
                kind,
            ),
            FairShareProfile::Model(model) if self.model_profiles.contains_key(model.as_str()) => {
                self.settle_model_reservation(
                    scope_id,
                    model,
                    tenant,
                    estimated_output_tokens,
                    observed_output_tokens,
                    kind,
                );
            }
            FairShareProfile::Model(_) => self.settle_global_reservation(
                scope_id,
                tenant,
                estimated_output_tokens,
                observed_output_tokens,
                kind,
            ),
        }
    }

    fn settle_global_reservation(
        &self,
        scope_id: u64,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
        observed_output_tokens: Option<u32>,
        kind: SettlementKind,
    ) {
        let charged = observed_output_tokens.unwrap_or(estimated_output_tokens);
        let mut state = self.state.lock();
        {
            let membership = state
                .scopes
                .entry(scope_id)
                .or_default()
                .tenants
                .entry(tenant.clone())
                .or_default();
            membership.active_reservations = membership.active_reservations.saturating_sub(1);
        }
        let system_virtual_time = state.system_virtual_time;
        let service = state.service.entry(tenant.clone()).or_default();
        service.active_reservations = service.active_reservations.saturating_sub(1);
        let correction =
            (f64::from(charged) - f64::from(estimated_output_tokens)) / self.weight(tenant);
        service.virtual_finish = (service.virtual_finish + correction).max(system_virtual_time);
        let virtual_service = service.virtual_finish;
        Self::advance_system_virtual_time(&mut state);
        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_sub(u64::from(estimated_output_tokens));
        accounting.charged_output_tokens = accounting
            .charged_output_tokens
            .saturating_add(u64::from(charged));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);

        let metric_tenant = self.metric_tenant(tenant);
        super::metrics::record_fair_share_charged_output_tokens(metric_tenant, charged);
        super::metrics::set_fair_share_virtual_finish("global", metric_tenant, virtual_service);
        super::metrics::set_fair_share_reserved_output_tokens(
            metric_tenant,
            reserved_output_tokens,
        );
        if kind != SettlementKind::Observed {
            super::metrics::record_fair_share_fallback(kind.as_str());
        }
    }

    fn settle_model_reservation(
        &self,
        scope_id: u64,
        model: &Arc<String>,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
        observed_output_tokens: Option<u32>,
        kind: SettlementKind,
    ) {
        let Some(policy) = self.model_profiles.get(model.as_str()) else {
            return;
        };
        let charged = observed_output_tokens.unwrap_or(estimated_output_tokens);
        let bucket = Self::model_bucket(policy, tenant);
        let is_other = bucket == ModelBucket::Other;
        let mut state = self.state.lock();
        let membership = state
            .scopes
            .entry(scope_id)
            .or_default()
            .model_tenants
            .entry(Arc::clone(model))
            .or_default()
            .entry(tenant.clone())
            .or_default();
        membership.active_reservations = membership.active_reservations.saturating_sub(1);

        let ledger = state.model_service.entry(Arc::clone(model)).or_default();
        let outer_weight = self.model_bucket_weight(policy, &bucket);
        let outer_service = ledger.outer.entries.entry(bucket.clone()).or_default();
        outer_service.active_reservations = outer_service.active_reservations.saturating_sub(1);
        let outer_correction =
            (f64::from(charged) - f64::from(estimated_output_tokens)) / outer_weight;
        outer_service.virtual_finish =
            (outer_service.virtual_finish + outer_correction).max(ledger.outer.system_virtual_time);
        let outer_virtual_finish = outer_service.virtual_finish;

        let inner_virtual_finish = if is_other {
            let service = ledger.other.entries.entry(tenant.clone()).or_default();
            service.active_reservations = service.active_reservations.saturating_sub(1);
            let correction =
                (f64::from(charged) - f64::from(estimated_output_tokens)) / self.default_weight;
            service.virtual_finish =
                (service.virtual_finish + correction).max(ledger.other.system_virtual_time);
            Some(service.virtual_finish)
        } else {
            None
        };
        Self::advance_service_clock(&mut ledger.outer);
        if is_other {
            Self::advance_service_clock(&mut ledger.other);
        }

        let accounting = state.accounting.entry(tenant.clone()).or_default();
        accounting.reserved_output_tokens = accounting
            .reserved_output_tokens
            .saturating_sub(u64::from(estimated_output_tokens));
        accounting.charged_output_tokens = accounting
            .charged_output_tokens
            .saturating_add(u64::from(charged));
        let reserved_output_tokens = accounting.reserved_output_tokens;
        drop(state);

        let metric_tenant = self.metric_tenant(tenant);
        super::metrics::record_fair_share_charged_output_tokens(metric_tenant, charged);
        super::metrics::set_fair_share_virtual_finish(
            model,
            metric_tenant,
            inner_virtual_finish.unwrap_or(outer_virtual_finish),
        );
        if is_other {
            super::metrics::set_fair_share_other_bucket_virtual_finish(model, outer_virtual_finish);
        }
        super::metrics::set_fair_share_reserved_output_tokens(
            metric_tenant,
            reserved_output_tokens,
        );
        if kind != SettlementKind::Observed {
            super::metrics::record_fair_share_fallback(kind.as_str());
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        scope_id: u64,
        tenant: &TenantKey,
    ) -> (u64, u64, u64, [usize; 4]) {
        let state = self.state.lock();
        let accounting = state
            .accounting
            .get(tenant)
            .expect("tenant accounting exists");
        let membership = state
            .scopes
            .get(&scope_id)
            .and_then(|scope| scope.tenants.get(tenant))
            .expect("tenant scope membership exists");
        (
            accounting.charged_output_tokens,
            accounting.reserved_output_tokens,
            membership.active_reservations,
            membership.queued,
        )
    }

    #[cfg(test)]
    pub(crate) fn model_snapshot(
        &self,
        scope_id: u64,
        model: &str,
        tenant: &TenantKey,
    ) -> (u64, u64, u64, [usize; 4]) {
        let state = self.state.lock();
        let accounting = state
            .accounting
            .get(tenant)
            .expect("tenant accounting exists");
        let membership = state
            .scopes
            .get(&scope_id)
            .and_then(|scope| {
                scope
                    .model_tenants
                    .iter()
                    .find(|(configured, _)| configured.as_str() == model)
                    .map(|(_, tenants)| tenants)
            })
            .and_then(|tenants| tenants.get(tenant))
            .expect("tenant model membership exists");
        (
            accounting.charged_output_tokens,
            accounting.reserved_output_tokens,
            membership.active_reservations,
            membership.queued,
        )
    }

    #[cfg(test)]
    fn virtual_snapshot(&self, scope_id: u64, tenant: &TenantKey) -> (f64, f64) {
        let state = self.state.lock();
        state
            .scopes
            .get(&scope_id)
            .and_then(|scope| scope.tenants.get(tenant))
            .expect("tenant scope membership exists");
        (
            state
                .service
                .get(tenant)
                .expect("tenant service exists")
                .virtual_finish,
            state.system_virtual_time,
        )
    }
}

impl SettlementKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::MissingUsage => "missing_usage",
            Self::Interrupted => "interrupted",
        }
    }
}

/// Provisional output-token charge attached to an admitted request.
///
/// Explicit settlement replaces the estimate with observed terminal usage.
/// Dropping without terminal usage keeps the estimate charged, preventing a
/// disconnect from becoming a way to escape fair-share accounting.
pub struct FairShareReservation {
    ledger: Arc<GlobalFairShare>,
    tenant: TenantKey,
    scope_id: u64,
    profile: FairShareProfile,
    estimated_output_tokens: u32,
    settled: bool,
}

impl std::fmt::Debug for FairShareReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FairShareReservation")
            .field("tenant", &self.tenant)
            .field("scope_id", &self.scope_id)
            .field("profile", &self.profile)
            .field("estimated_output_tokens", &self.estimated_output_tokens)
            .field("settled", &self.settled)
            .finish()
    }
}

impl FairShareReservation {
    pub(crate) fn tenant(&self) -> &TenantKey {
        &self.tenant
    }

    pub fn settle(mut self, observed_output_tokens: Option<u32>, kind: SettlementKind) {
        self.ledger.settle_reservation(
            self.scope_id,
            &self.profile,
            &self.tenant,
            self.estimated_output_tokens,
            observed_output_tokens,
            kind,
        );
        self.settled = true;
    }

    pub fn cancel(mut self) {
        self.ledger.cancel_reservation(
            self.scope_id,
            &self.profile,
            &self.tenant,
            self.estimated_output_tokens,
        );
        self.settled = true;
    }
}

impl Drop for FairShareReservation {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        self.ledger.settle_reservation(
            self.scope_id,
            &self.profile,
            &self.tenant,
            self.estimated_output_tokens,
            None,
            SettlementKind::Interrupted,
        );
        self.settled = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::scheduler::ModelFairShareConfig;

    fn config(weights: &[(&str, f64)]) -> FairShareConfig {
        FairShareConfig {
            default_weight: 1.0,
            default_output_tokens: 10,
            trust_output_token_estimate_header: false,
            trust_request_model_header: false,
            tenant_weights: weights
                .iter()
                .map(|(tenant, weight)| ((*tenant).to_string(), *weight))
                .collect(),
            model_profiles: HashMap::new(),
        }
    }

    fn reserve_one(
        ledger: &Arc<GlobalFairShare>,
        scope_id: u64,
        class: Class,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
    ) -> Option<FairShareReservation> {
        ledger.register_waiter(scope_id, tenant, class);
        ledger
            .reserve_local_candidate(
                scope_id,
                class,
                &[FairShareCandidate {
                    index: 0,
                    tenant,
                    estimated_output_tokens,
                }],
            )
            .map(|selection| selection.reservation)
    }

    fn model_config() -> FairShareConfig {
        let mut config = config(&[]);
        config.trust_request_model_header = true;
        config.model_profiles = HashMap::from([
            (
                "deepseek-v4-flash".to_string(),
                ModelFairShareConfig {
                    tenant_weights: HashMap::from([
                        ("header:junu".to_string(), 30.0),
                        ("header:xuezhou".to_string(), 40.0),
                        ("header:zhenting".to_string(), 10.0),
                    ]),
                    other_weight: 20.0,
                },
            ),
            (
                "kimi-k3".to_string(),
                ModelFairShareConfig {
                    tenant_weights: HashMap::from([("header:mukhesh".to_string(), 80.0)]),
                    other_weight: 20.0,
                },
            ),
        ]);
        config
    }

    fn reserve_one_in_profile(
        ledger: &Arc<GlobalFairShare>,
        scope_id: u64,
        profile: &FairShareProfile,
        class: Class,
        tenant: &TenantKey,
        estimated_output_tokens: u32,
    ) -> Option<FairShareReservation> {
        ledger.register_waiter_in_profile(scope_id, profile, tenant, class);
        ledger
            .reserve_local_candidate_in_profile(
                scope_id,
                profile,
                class,
                &[FairShareCandidate {
                    index: 0,
                    tenant,
                    estimated_output_tokens,
                }],
            )
            .map(|selection| selection.reservation)
    }

    #[test]
    fn observed_tokens_replace_the_provisional_estimate_once() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[("header:a", 1.0)])));
        let scope_id = ledger.new_scope();
        let tenant = TenantKey::new("header:a");
        let reservation = reserve_one(&ledger, scope_id, Class::Default, &tenant, 100).unwrap();
        assert_eq!(ledger.snapshot(scope_id, &tenant), (0, 100, 1, [0; 4]));

        reservation.settle(Some(37), SettlementKind::Observed);
        assert_eq!(ledger.snapshot(scope_id, &tenant), (37, 0, 0, [0; 4]));
    }

    #[test]
    fn cancelled_delivery_reverts_the_provisional_charge() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[("header:a", 1.0)])));
        let scope_id = ledger.new_scope();
        let tenant = TenantKey::new("header:a");
        let reservation = reserve_one(&ledger, scope_id, Class::Default, &tenant, 100).unwrap();
        reservation.cancel();
        assert_eq!(ledger.snapshot(scope_id, &tenant), (0, 0, 0, [0; 4]));
    }

    #[test]
    fn missing_terminal_usage_keeps_the_estimate_charged() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[("header:a", 1.0)])));
        let scope_id = ledger.new_scope();
        let tenant = TenantKey::new("header:a");
        let reservation = reserve_one(&ledger, scope_id, Class::Default, &tenant, 100).unwrap();
        drop(reservation);
        assert_eq!(ledger.snapshot(scope_id, &tenant), (100, 0, 0, [0; 4]));
    }

    #[test]
    fn weighted_share_converges_when_users_contend_for_one_pool() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 2.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");
        let mut admitted_a = 0;
        let mut admitted_b = 0;

        // Keep both tenants continuously backlogged for every selection.
        for _ in 0..30 {
            ledger.register_waiter(scope_id, &a, Class::Default);
            ledger.register_waiter(scope_id, &b, Class::Default);
        }
        for _ in 0..30 {
            let candidates = [
                FairShareCandidate {
                    index: 0,
                    tenant: &a,
                    estimated_output_tokens: 10,
                },
                FairShareCandidate {
                    index: 1,
                    tenant: &b,
                    estimated_output_tokens: 10,
                },
            ];
            let selection = ledger
                .reserve_local_candidate(scope_id, Class::Default, &candidates)
                .expect("a local contender must be selected");
            if selection.index == 0 {
                admitted_a += 1;
                selection
                    .reservation
                    .settle(Some(10), SettlementKind::Observed);
            } else {
                admitted_b += 1;
                selection
                    .reservation
                    .settle(Some(10), SettlementKind::Observed);
            }
        }

        assert_eq!((admitted_a, admitted_b), (20, 10));
    }

    #[test]
    fn lone_backlogged_user_borrows_all_available_capacity() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");

        for _ in 0..4 {
            reserve_one(&ledger, scope_id, Class::Default, &a, 10)
                .expect("a lone eligible user must never be idled")
                .settle(Some(10), SettlementKind::Observed);
        }
        assert_eq!(ledger.snapshot(scope_id, &a).0, 40);
    }

    #[test]
    fn underserved_eligible_user_wins_under_simultaneous_contention() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        // b is already backlogged while a receives service, so this is real
        // debt on a shared constrained resource rather than idle-time credit.
        ledger.register_waiter(scope_id, &a, Class::Default);
        ledger.register_waiter(scope_id, &b, Class::Default);
        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let first = ledger
            .reserve_local_candidate(scope_id, Class::Default, &candidates)
            .unwrap();
        assert_eq!(first.index, 0, "stable tie-break selects a first");
        first.reservation.settle(Some(50), SettlementKind::Observed);
        ledger.register_waiter(scope_id, &a, Class::Default);

        let selected = ledger
            .reserve_local_candidate(scope_id, Class::Default, &candidates)
            .unwrap();
        assert_eq!(selected.index, 1, "the underserved eligible user wins");
        selected.reservation.cancel();
    }

    #[test]
    fn solo_service_does_not_create_debt_against_an_idle_tenant() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        for _ in 0..100 {
            reserve_one(&ledger, scope_id, Class::Default, &a, 10)
                .unwrap()
                .settle(Some(10), SettlementKind::Observed);
        }
        ledger.register_waiter(scope_id, &a, Class::Default);
        ledger.register_waiter(scope_id, &b, Class::Default);

        let (a_finish, system_time) = ledger.virtual_snapshot(scope_id, &a);
        let (b_finish, _) = ledger.virtual_snapshot(scope_id, &b);
        assert_eq!(a_finish, system_time);
        assert_eq!(b_finish, system_time);

        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let selected = ledger
            .reserve_local_candidate(scope_id, Class::Default, &candidates)
            .unwrap();
        assert_eq!(
            selected.index, 0,
            "new b must not monopolize service to match a's lifetime total"
        );
        selected.reservation.cancel();
    }

    #[test]
    fn idle_returning_tenant_is_prompt_without_unbounded_catch_up() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        for _ in 0..20 {
            reserve_one(&ledger, scope_id, Class::Default, &a, 10)
                .unwrap()
                .settle(Some(10), SettlementKind::Observed);
        }
        for _ in 0..4 {
            ledger.register_waiter(scope_id, &a, Class::Default);
            ledger.register_waiter(scope_id, &b, Class::Default);
        }
        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let mut order = Vec::new();
        for _ in 0..4 {
            let selected = ledger
                .reserve_local_candidate(scope_id, Class::Default, &candidates)
                .unwrap();
            order.push(selected.index);
            selected
                .reservation
                .settle(Some(10), SettlementKind::Observed);
        }
        assert_eq!(order, vec![0, 1, 0, 1]);
    }

    #[test]
    fn unrelated_model_backlog_never_blocks_local_capacity() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let local_scope = ledger.new_scope();
        let other_scope = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        // b is queued only for another model. It cannot consume this scope's
        // slot or cause local idling.
        ledger.register_waiter(other_scope, &b, Class::Default);
        ledger.register_waiter(local_scope, &a, Class::Default);
        let selected = ledger
            .reserve_local_candidate(
                local_scope,
                Class::Default,
                &[FairShareCandidate {
                    index: 0,
                    tenant: &a,
                    estimated_output_tokens: 10,
                }],
            )
            .expect("an eligible local request must keep the pool work-conserving");
        selected.reservation.cancel();
    }

    #[test]
    fn contended_service_in_one_pool_affects_ordering_in_another_pool() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let first_pool = ledger.new_scope();
        let second_pool = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        // b is backlogged while a receives service in the first pool, so a's
        // canonical process-global virtual finish moves ahead of b's.
        ledger.register_waiter(first_pool, &b, Class::Default);
        reserve_one(&ledger, first_pool, Class::Default, &a, 50)
            .unwrap()
            .settle(Some(50), SettlementKind::Observed);

        ledger.register_waiter(second_pool, &a, Class::Default);
        ledger.register_waiter(second_pool, &b, Class::Default);
        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let selected = ledger
            .reserve_local_candidate(second_pool, Class::Default, &candidates)
            .unwrap();
        assert_eq!(
            selected.index, 1,
            "contended service debt must follow a tenant across model pools"
        );
        selected.reservation.cancel();
        assert_eq!(ledger.snapshot(first_pool, &a).0, 50);
    }

    #[test]
    fn batch_and_interactive_share_one_partition_ledger() {
        let ledger = Arc::new(GlobalFairShare::from_config(&config(&[
            ("header:a", 1.0),
            ("header:b", 1.0),
        ])));
        let scope_id = ledger.new_scope();
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        // b is backlogged in the partition while a receives bulk service.
        // Class priority remains the outer policy, but the tenant's virtual
        // finish is shared when a later interactive contention is resolved.
        ledger.register_waiter(scope_id, &b, Class::Default);
        let bulk = reserve_one(&ledger, scope_id, Class::Bulk, &a, 50).unwrap();
        bulk.settle(Some(50), SettlementKind::Observed);

        ledger.register_waiter(scope_id, &a, Class::Interactive);
        ledger.register_waiter(scope_id, &b, Class::Interactive);
        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let selected = ledger
            .reserve_local_candidate(scope_id, Class::Interactive, &candidates)
            .unwrap();
        assert_eq!(
            selected.index, 1,
            "bulk output remains visible to later interactive contention"
        );
        selected.reservation.cancel();
        assert_eq!(ledger.snapshot(scope_id, &a).0, 50);
    }

    #[test]
    fn deepseek_profile_enforces_named_shares_and_one_aggregate_other_bucket() {
        let ledger = Arc::new(GlobalFairShare::from_config(&model_config()));
        let scope_id = ledger.new_scope();
        let profile = ledger.profile_for_model("deepseek-v4-flash");
        let tenants = [
            TenantKey::new("header:junu"),
            TenantKey::new("header:xuezhou"),
            TenantKey::new("header:zhenting"),
            TenantKey::new("header:other-a"),
            TenantKey::new("header:other-b"),
        ];
        for tenant in &tenants {
            for _ in 0..100 {
                ledger.register_waiter_in_profile(scope_id, &profile, tenant, Class::Default);
            }
        }
        let mut admitted = [0_u32; 5];
        for _ in 0..100 {
            let candidates: Vec<_> = tenants
                .iter()
                .enumerate()
                .map(|(index, tenant)| FairShareCandidate {
                    index,
                    tenant,
                    estimated_output_tokens: 10,
                })
                .collect();
            let selection = ledger
                .reserve_local_candidate_in_profile(scope_id, &profile, Class::Default, &candidates)
                .expect("a deepseek contender must be selected");
            admitted[selection.index] += 1;
            selection
                .reservation
                .settle(Some(10), SettlementKind::Observed);
        }

        assert_eq!(admitted, [30, 40, 10, 10, 10]);
        assert_eq!(
            admitted[3] + admitted[4],
            20,
            "all unlisted users share one fixed 20% outer bucket"
        );
    }

    #[test]
    fn kimi_profile_gives_mukhesh_eighty_and_other_users_twenty() {
        let ledger = Arc::new(GlobalFairShare::from_config(&model_config()));
        let scope_id = ledger.new_scope();
        let profile = ledger.profile_for_model("kimi-k3");
        let tenants = [
            TenantKey::new("header:mukhesh"),
            TenantKey::new("header:other-a"),
            TenantKey::new("header:other-b"),
        ];
        for tenant in &tenants {
            for _ in 0..100 {
                ledger.register_waiter_in_profile(scope_id, &profile, tenant, Class::Default);
            }
        }
        let mut admitted = [0_u32; 3];
        for _ in 0..100 {
            let candidates: Vec<_> = tenants
                .iter()
                .enumerate()
                .map(|(index, tenant)| FairShareCandidate {
                    index,
                    tenant,
                    estimated_output_tokens: 10,
                })
                .collect();
            let selection = ledger
                .reserve_local_candidate_in_profile(scope_id, &profile, Class::Default, &candidates)
                .expect("a kimi contender must be selected");
            admitted[selection.index] += 1;
            selection
                .reservation
                .settle(Some(10), SettlementKind::Observed);
        }

        assert_eq!(admitted, [80, 10, 10]);
    }

    #[test]
    fn model_scoped_debt_is_independent_but_accounting_is_process_global() {
        let mut config = config(&[]);
        config.trust_request_model_header = true;
        let equal = ModelFairShareConfig {
            tenant_weights: HashMap::from([
                ("header:a".to_string(), 1.0),
                ("header:b".to_string(), 1.0),
            ]),
            other_weight: 1.0,
        };
        config.model_profiles = HashMap::from([
            ("deepseek".to_string(), equal.clone()),
            ("kimi".to_string(), equal),
        ]);
        let ledger = Arc::new(GlobalFairShare::from_config(&config));
        let scope_id = ledger.new_scope();
        let deepseek = ledger.profile_for_model("deepseek");
        let kimi = ledger.profile_for_model("kimi");
        let a = TenantKey::new("header:a");
        let b = TenantKey::new("header:b");

        ledger.register_waiter_in_profile(scope_id, &deepseek, &b, Class::Default);
        reserve_one_in_profile(&ledger, scope_id, &deepseek, Class::Default, &a, 50)
            .unwrap()
            .settle(Some(50), SettlementKind::Observed);

        ledger.register_waiter_in_profile(scope_id, &kimi, &a, Class::Default);
        ledger.register_waiter_in_profile(scope_id, &kimi, &b, Class::Default);
        let candidates = [
            FairShareCandidate {
                index: 0,
                tenant: &a,
                estimated_output_tokens: 10,
            },
            FairShareCandidate {
                index: 1,
                tenant: &b,
                estimated_output_tokens: 10,
            },
        ];
        let selected = ledger
            .reserve_local_candidate_in_profile(scope_id, &kimi, Class::Default, &candidates)
            .unwrap();
        assert_eq!(
            selected.index, 0,
            "deepseek debt must not bias an independent kimi profile"
        );
        selected
            .reservation
            .settle(Some(10), SettlementKind::Observed);
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi", &a).0,
            60,
            "charged output accounting remains process-global across models"
        );
    }

    #[test]
    fn model_reservation_cancel_and_missing_usage_keep_existing_semantics() {
        let ledger = Arc::new(GlobalFairShare::from_config(&model_config()));
        let scope_id = ledger.new_scope();
        let profile = ledger.profile_for_model("kimi-k3");
        let tenant = TenantKey::new("header:other-a");

        reserve_one_in_profile(&ledger, scope_id, &profile, Class::Default, &tenant, 100)
            .unwrap()
            .cancel();
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (0, 0, 0, [0; 4])
        );

        drop(
            reserve_one_in_profile(&ledger, scope_id, &profile, Class::Default, &tenant, 100)
                .unwrap(),
        );
        assert_eq!(
            ledger.model_snapshot(scope_id, "kimi-k3", &tenant),
            (100, 0, 0, [0; 4])
        );
    }
}
