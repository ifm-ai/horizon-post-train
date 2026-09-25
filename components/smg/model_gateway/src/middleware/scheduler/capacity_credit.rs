//! Short-lived, single-use capacity credit registry.
//!
//! An issued credit owns one scheduler permit and its provisional fair-share
//! reservation until the matching inference request redeems it. Cancellation
//! or expiry releases both resources. The registry remains transport-agnostic;
//! [`super::capacity_credit_api`] provides the authenticated internal HTTP
//! issue/cancel surface and validates inference-request bindings.

use std::{
    collections::{btree_map::Entry, BTreeMap, HashMap},
    fmt,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use parking_lot::Mutex;
use rand::Rng;
use thiserror::Error;

use super::{metrics as sched_metrics, SchedulerPermit};
use crate::tenant::TenantKey;

const TOKEN_BYTES: usize = 32;
const MAX_DISPATCH_ID_BYTES: usize = 128;

/// Scheduler slot held between credit issue and redemption. If the credit is
/// cancelled or expires before backend work starts, Drop cancels the
/// provisional fair-share reservation before releasing the slot.
pub struct HeldSchedulerPermit {
    permit: Option<SchedulerPermit>,
}

impl HeldSchedulerPermit {
    #[must_use]
    pub fn new(permit: SchedulerPermit) -> Self {
        Self {
            permit: Some(permit),
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "the owned payload can be extracted only by consuming this wrapper"
    )]
    pub fn into_permit(mut self) -> SchedulerPermit {
        self.permit
            .take()
            .expect("held scheduler permit can be consumed only once")
    }

    /// Bind a successfully redeemed credit to the held scheduler permit.
    /// Returns false rather than minting a proof if the permit has no matching
    /// fair-share reservation.
    pub(crate) fn bind_redeemed_capacity_credit(
        &mut self,
        binding: &CapacityCreditBinding,
        route_request_id: uuid::Uuid,
    ) -> bool {
        self.permit.as_mut().is_some_and(|permit| {
            permit.attach_redeemed_capacity_credit_proof(binding, route_request_id)
        })
    }
}

impl fmt::Debug for HeldSchedulerPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeldSchedulerPermit")
            .field("active", &self.permit.is_some())
            .finish()
    }
}

impl Drop for HeldSchedulerPermit {
    fn drop(&mut self) {
        if let Some(mut permit) = self.permit.take() {
            permit.cancel_fair_share_reservation();
            drop(permit);
        }
    }
}

/// Opaque bearer token for one capacity credit.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CapacityCreditToken([u8; TOKEN_BYTES]);

impl CapacityCreditToken {
    fn random() -> Self {
        let mut bytes = [0_u8; TOKEN_BYTES];
        rand::rng().fill_bytes(&mut bytes);
        Self(bytes)
    }

    /// Encode the token for an internal HTTP header.
    #[must_use]
    pub fn expose_secret(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.0)
    }

    /// Parse an internal header value into a fixed-size token.
    pub fn parse(encoded: &str) -> Result<Self, CapacityCreditError> {
        let decoded = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| CapacityCreditError::InvalidToken)?;
        let bytes = decoded
            .try_into()
            .map_err(|_| CapacityCreditError::InvalidToken)?;
        Ok(Self(bytes))
    }
}

impl fmt::Debug for CapacityCreditToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CapacityCreditToken([REDACTED])")
    }
}

/// Immutable identity and accounting scope of one credit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityCreditBinding {
    generation: Arc<str>,
    policy_epoch: u64,
    partition: Arc<str>,
    model: Arc<str>,
    tenant: TenantKey,
    request_id: Arc<str>,
    estimated_output_tokens: u32,
}

impl CapacityCreditBinding {
    /// Build a validated binding for one allocator-selected request attempt.
    pub fn new(
        generation: impl AsRef<str>,
        policy_epoch: u64,
        partition: impl AsRef<str>,
        model: impl AsRef<str>,
        tenant: TenantKey,
        request_id: impl AsRef<str>,
        estimated_output_tokens: u32,
    ) -> Result<Self, CapacityCreditError> {
        let generation = checked_label("generation", generation.as_ref())?;
        let partition = checked_label("partition", partition.as_ref())?;
        let model = checked_label("model", model.as_ref())?;
        let request_id = checked_label("request id", request_id.as_ref())?;
        if tenant.as_str().is_empty() || tenant.as_str().trim() != tenant.as_str() {
            return Err(CapacityCreditError::InvalidBinding("tenant"));
        }
        if estimated_output_tokens == 0 {
            return Err(CapacityCreditError::InvalidBinding(
                "estimated output tokens",
            ));
        }
        Ok(Self {
            generation,
            policy_epoch,
            partition,
            model,
            tenant,
            request_id,
            estimated_output_tokens,
        })
    }

    /// Allocator generation allowed to use this credit.
    #[must_use]
    pub fn generation(&self) -> &str {
        &self.generation
    }

    /// Logical request-attempt identifier used for idempotent issue.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    #[must_use]
    pub fn policy_epoch(&self) -> u64 {
        self.policy_epoch
    }

    #[must_use]
    pub fn partition(&self) -> &str {
        &self.partition
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[must_use]
    pub fn tenant(&self) -> &TenantKey {
        &self.tenant
    }

    #[must_use]
    pub fn estimated_output_tokens(&self) -> u32 {
        self.estimated_output_tokens
    }
}

fn checked_label(name: &'static str, value: &str) -> Result<Arc<str>, CapacityCreditError> {
    if value.is_empty() || value.trim() != value {
        return Err(CapacityCreditError::InvalidBinding(name));
    }
    Ok(Arc::from(value))
}

fn checked_dispatch_id(value: &str) -> Result<Arc<str>, CapacityCreditError> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > MAX_DISPATCH_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(CapacityCreditError::InvalidBinding("dispatch id"));
    }
    Ok(Arc::from(value))
}

/// Result of issuing a new or idempotently repeated active credit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IssuedCapacityCredit {
    pub token: CapacityCreditToken,
    pub expires_at: Instant,
    pub newly_issued: bool,
}

/// Result of starting, or idempotently retrying, the bounded presentation
/// window for an active unredeemed credit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmedCapacityCredit {
    pub expires_at: Instant,
    pub newly_armed: bool,
}

/// Successfully redeemed credit and its caller-owned capacity payload.
#[derive(Debug)]
pub struct RedeemedCapacityCredit<P> {
    pub binding: CapacityCreditBinding,
    pub payload: P,
}

/// Capacity-credit protocol failure.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CapacityCreditError {
    #[error("invalid capacity credit token")]
    InvalidToken,
    #[error("invalid capacity credit binding field: {0}")]
    InvalidBinding(&'static str),
    #[error("capacity credit belongs to another allocator generation")]
    WrongGeneration,
    #[error("request id already has a credit with different bindings")]
    RequestConflict,
    #[error("capacity credit is unknown")]
    Unknown,
    #[error("capacity credit binding does not match")]
    BindingMismatch,
    #[error("capacity credit expired")]
    Expired,
    #[error("capacity credit was already redeemed")]
    AlreadyRedeemed,
    #[error("capacity credit was armed by another dispatch")]
    AlreadyArmed,
    #[error("capacity credit was cancelled")]
    Cancelled,
    #[error("capacity credit registry state is inconsistent")]
    InternalState,
    #[error("scheduler has no capacity for a new credit")]
    NoCapacity,
}

enum CreditState<P> {
    Active(P),
    Redeemed,
    Cancelled,
    Expired,
}

struct CreditRecord<P> {
    binding: CapacityCreditBinding,
    expires_at: Instant,
    armed_by: Option<Arc<str>>,
    terminal_until: Option<Instant>,
    state: CreditState<P>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeadlineKind {
    Active,
    Terminal,
}

struct RegistryInner<P> {
    by_token: HashMap<CapacityCreditToken, CreditRecord<P>>,
    by_request: HashMap<Arc<str>, CapacityCreditToken>,
    deadlines: BTreeMap<(Instant, u64), (CapacityCreditToken, DeadlineKind)>,
    next_deadline_id: u64,
}

/// Process-local registry for one active SMG generation.
pub struct CapacityCreditRegistry<P> {
    generation: Arc<str>,
    ttl: Duration,
    terminal_retention: Duration,
    inner: Mutex<RegistryInner<P>>,
}

impl<P> CapacityCreditRegistry<P> {
    /// Create a registry with a nonzero active TTL and replay-tombstone TTL.
    pub fn new(
        generation: impl AsRef<str>,
        ttl: Duration,
        terminal_retention: Duration,
    ) -> Result<Self, CapacityCreditError> {
        let generation = checked_label("generation", generation.as_ref())?;
        if ttl.is_zero() {
            return Err(CapacityCreditError::InvalidBinding("credit ttl"));
        }
        if terminal_retention.is_zero() {
            return Err(CapacityCreditError::InvalidBinding("terminal retention"));
        }
        Ok(Self {
            generation,
            ttl,
            terminal_retention,
            inner: Mutex::new(RegistryInner {
                by_token: HashMap::new(),
                by_request: HashMap::new(),
                deadlines: BTreeMap::new(),
                next_deadline_id: 0,
            }),
        })
    }

    /// Issue a credit, or return the same active token for an exact retry.
    pub fn issue(
        &self,
        binding: CapacityCreditBinding,
        payload: P,
    ) -> Result<IssuedCapacityCredit, CapacityCreditError> {
        self.issue_with_at(binding, || Ok(payload), Instant::now())
    }

    /// Issue a credit while constructing its capacity payload only when the
    /// request does not already own an active credit. The factory must be
    /// immediate and nonblocking because it runs under the registry lock.
    pub fn issue_with<F>(
        &self,
        binding: CapacityCreditBinding,
        payload_factory: F,
    ) -> Result<IssuedCapacityCredit, CapacityCreditError>
    where
        F: FnOnce() -> Result<P, CapacityCreditError>,
    {
        self.issue_with_at(binding, payload_factory, Instant::now())
    }

    #[cfg(test)]
    fn issue_at(
        &self,
        binding: CapacityCreditBinding,
        payload: P,
        now: Instant,
    ) -> Result<IssuedCapacityCredit, CapacityCreditError> {
        self.issue_with_at(binding, || Ok(payload), now)
    }

    fn issue_with_at<F>(
        &self,
        binding: CapacityCreditBinding,
        payload_factory: F,
        now: Instant,
    ) -> Result<IssuedCapacityCredit, CapacityCreditError>
    where
        F: FnOnce() -> Result<P, CapacityCreditError>,
    {
        let metric_partition = Arc::clone(&binding.partition);
        if binding.generation.as_ref() != self.generation.as_ref() {
            sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::WRONG_GENERATION,
            );
            return Err(CapacityCreditError::WrongGeneration);
        }
        self.maintain_at(now);

        let mut unused_factory = Some(payload_factory);
        let result = {
            let mut inner = self.inner.lock();
            if let Some(token) = inner.by_request.get(&binding.request_id).cloned() {
                let Some(record) = inner.by_token.get(&token) else {
                    return Err(CapacityCreditError::InternalState);
                };
                if record.binding == binding {
                    match record.state {
                        CreditState::Active(_) => Ok(IssuedCapacityCredit {
                            token,
                            expires_at: record.expires_at,
                            newly_issued: false,
                        }),
                        CreditState::Redeemed => Err(CapacityCreditError::AlreadyRedeemed),
                        CreditState::Cancelled => Err(CapacityCreditError::Cancelled),
                        CreditState::Expired => Err(CapacityCreditError::Expired),
                    }
                } else {
                    Err(CapacityCreditError::RequestConflict)
                }
            } else {
                let factory = unused_factory
                    .take()
                    .ok_or(CapacityCreditError::InternalState)?;
                let payload = factory()?;
                let token = loop {
                    let candidate = CapacityCreditToken::random();
                    if !inner.by_token.contains_key(&candidate) {
                        break candidate;
                    }
                };
                let expires_at = now + self.ttl;
                schedule_deadline(&mut inner, expires_at, token.clone(), DeadlineKind::Active);
                inner
                    .by_request
                    .insert(Arc::clone(&binding.request_id), token.clone());
                inner.by_token.insert(
                    token.clone(),
                    CreditRecord {
                        binding,
                        expires_at,
                        armed_by: None,
                        terminal_until: None,
                        state: CreditState::Active(payload),
                    },
                );
                Ok(IssuedCapacityCredit {
                    token,
                    expires_at,
                    newly_issued: true,
                })
            }
        };
        drop(unused_factory);
        match &result {
            Ok(issued) if issued.newly_issued => {
                sched_metrics::record_capacity_credit_operation(
                    &metric_partition,
                    sched_metrics::capacity_credit_outcome::ISSUED,
                );
                sched_metrics::increment_capacity_credit_active(&metric_partition);
            }
            Ok(_) => sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::IDEMPOTENT_RETRY,
            ),
            Err(error) => record_capacity_credit_error(&metric_partition, *error),
        }
        result
    }

    /// Start the bounded presentation window for a matching active credit.
    ///
    /// The first successful call replaces the issue-time deadline with one
    /// fresh TTL while leaving the exact payload and immutable binding in
    /// place. Exact retries return the same deadline and cannot extend it.
    pub fn arm(
        &self,
        token: &CapacityCreditToken,
        presented: &CapacityCreditBinding,
        dispatch_id: &str,
    ) -> Result<ArmedCapacityCredit, CapacityCreditError> {
        self.arm_at(token, presented, dispatch_id, Instant::now())
    }

    fn arm_at(
        &self,
        token: &CapacityCreditToken,
        presented: &CapacityCreditBinding,
        dispatch_id: &str,
        now: Instant,
    ) -> Result<ArmedCapacityCredit, CapacityCreditError> {
        let metric_partition = Arc::clone(&presented.partition);
        let dispatch_id = match checked_dispatch_id(dispatch_id) {
            Ok(dispatch_id) => dispatch_id,
            Err(error) => {
                sched_metrics::record_capacity_credit_operation(
                    &metric_partition,
                    sched_metrics::capacity_credit_outcome::ARM_REJECTED,
                );
                return Err(error);
            }
        };
        if presented.generation.as_ref() != self.generation.as_ref() {
            sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::ARM_REJECTED,
            );
            return Err(CapacityCreditError::WrongGeneration);
        }

        let (result, expired_payload, metric_partition) = {
            let mut inner = self.inner.lock();
            let Some(record) = inner.by_token.get_mut(token) else {
                sched_metrics::record_capacity_credit_operation(
                    "unknown",
                    sched_metrics::capacity_credit_outcome::ARM_REJECTED,
                );
                return Err(CapacityCreditError::Unknown);
            };
            let metric_partition = Arc::clone(&record.binding.partition);
            let terminal_until = now + self.terminal_retention;
            let mut expired_payload = None;
            let result = if record.binding == *presented {
                match record.state {
                    CreditState::Active(_) if now >= record.expires_at => {
                        expired_payload =
                            transition_terminal(record, CreditState::Expired, terminal_until);
                        if expired_payload.is_some() {
                            schedule_deadline(
                                &mut inner,
                                terminal_until,
                                token.clone(),
                                DeadlineKind::Terminal,
                            );
                        }
                        Err(CapacityCreditError::Expired)
                    }
                    CreditState::Active(_)
                        if record.armed_by.as_deref() == Some(dispatch_id.as_ref()) =>
                    {
                        Ok(ArmedCapacityCredit {
                            expires_at: record.expires_at,
                            newly_armed: false,
                        })
                    }
                    CreditState::Active(_) if record.armed_by.is_some() => {
                        Err(CapacityCreditError::AlreadyArmed)
                    }
                    CreditState::Active(_) => {
                        let expires_at = now + self.ttl;
                        record.expires_at = expires_at;
                        record.armed_by = Some(dispatch_id);
                        schedule_deadline(
                            &mut inner,
                            expires_at,
                            token.clone(),
                            DeadlineKind::Active,
                        );
                        Ok(ArmedCapacityCredit {
                            expires_at,
                            newly_armed: true,
                        })
                    }
                    CreditState::Redeemed => Err(CapacityCreditError::AlreadyRedeemed),
                    CreditState::Cancelled => Err(CapacityCreditError::Cancelled),
                    CreditState::Expired => Err(CapacityCreditError::Expired),
                }
            } else {
                Err(CapacityCreditError::BindingMismatch)
            };
            (result, expired_payload, metric_partition)
        };

        if expired_payload.is_some() {
            sched_metrics::decrement_capacity_credit_active(&metric_partition);
        }
        match &result {
            Ok(armed) if armed.newly_armed => {
                sched_metrics::record_capacity_credit_operation(
                    &metric_partition,
                    sched_metrics::capacity_credit_outcome::ARMED,
                );
            }
            Ok(_) => sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::ARM_IDEMPOTENT_RETRY,
            ),
            Err(_) => sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::ARM_REJECTED,
            ),
        }
        drop(expired_payload);
        result
    }

    /// Atomically consume a matching active credit at most once.
    pub fn redeem(
        &self,
        token: &CapacityCreditToken,
        presented: &CapacityCreditBinding,
    ) -> Result<RedeemedCapacityCredit<P>, CapacityCreditError> {
        self.redeem_at(token, presented, Instant::now())
    }

    fn redeem_at(
        &self,
        token: &CapacityCreditToken,
        presented: &CapacityCreditBinding,
        now: Instant,
    ) -> Result<RedeemedCapacityCredit<P>, CapacityCreditError> {
        let (result, expired_payload, metric_partition, active_released) = {
            let mut inner = self.inner.lock();
            let Some(record) = inner.by_token.get_mut(token) else {
                sched_metrics::record_capacity_credit_operation(
                    "unknown",
                    sched_metrics::capacity_credit_outcome::UNKNOWN,
                );
                return Err(CapacityCreditError::Unknown);
            };
            let metric_partition = Arc::clone(&record.binding.partition);
            let terminal_until = now + self.terminal_retention;
            if now >= record.expires_at {
                let payload = transition_terminal(record, CreditState::Expired, terminal_until);
                let active_released = payload.is_some();
                let result = (Err(CapacityCreditError::Expired), payload);
                if result.1.is_some() {
                    schedule_deadline(
                        &mut inner,
                        terminal_until,
                        token.clone(),
                        DeadlineKind::Terminal,
                    );
                }
                (result.0, result.1, metric_partition, active_released)
            } else if record.binding != *presented {
                (
                    Err(CapacityCreditError::BindingMismatch),
                    None,
                    metric_partition,
                    false,
                )
            } else {
                let (result, active_released) =
                    match std::mem::replace(&mut record.state, CreditState::Redeemed) {
                        CreditState::Active(payload) => {
                            record.terminal_until = Some(terminal_until);
                            let result = (
                                Ok(RedeemedCapacityCredit {
                                    binding: record.binding.clone(),
                                    payload,
                                }),
                                None,
                            );
                            schedule_deadline(
                                &mut inner,
                                terminal_until,
                                token.clone(),
                                DeadlineKind::Terminal,
                            );
                            (result, true)
                        }
                        CreditState::Redeemed => {
                            record.state = CreditState::Redeemed;
                            ((Err(CapacityCreditError::AlreadyRedeemed), None), false)
                        }
                        CreditState::Cancelled => {
                            record.state = CreditState::Cancelled;
                            ((Err(CapacityCreditError::Cancelled), None), false)
                        }
                        CreditState::Expired => {
                            record.state = CreditState::Expired;
                            ((Err(CapacityCreditError::Expired), None), false)
                        }
                    };
                (result.0, result.1, metric_partition, active_released)
            }
        };
        if active_released {
            sched_metrics::decrement_capacity_credit_active(&metric_partition);
        }
        match &result {
            Ok(_) => sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::REDEEMED,
            ),
            Err(error) => record_capacity_credit_error(&metric_partition, *error),
        }
        drop(expired_payload);
        result
    }

    /// Cancel an active credit. Returns `true` only for the first transition.
    pub fn cancel(&self, token: &CapacityCreditToken) -> bool {
        self.cancel_at(token, Instant::now())
    }

    fn cancel_at(&self, token: &CapacityCreditToken, now: Instant) -> bool {
        let terminal_until = now + self.terminal_retention;
        let (payload, metric_partition) = {
            let mut inner = self.inner.lock();
            let Some(record) = inner.by_token.get_mut(token) else {
                return false;
            };
            let metric_partition = Arc::clone(&record.binding.partition);
            let payload = transition_terminal(record, CreditState::Cancelled, terminal_until);
            if payload.is_some() {
                schedule_deadline(
                    &mut inner,
                    terminal_until,
                    token.clone(),
                    DeadlineKind::Terminal,
                );
            }
            (payload, metric_partition)
        };
        let changed = payload.is_some();
        if changed {
            sched_metrics::record_capacity_credit_operation(
                &metric_partition,
                sched_metrics::capacity_credit_outcome::CANCELLED,
            );
            sched_metrics::decrement_capacity_credit_active(&metric_partition);
        }
        drop(payload);
        changed
    }

    /// Expire active credits and purge elapsed replay tombstones.
    pub fn maintain(&self) -> usize {
        self.maintain_at(Instant::now())
    }

    /// Reap expired credits without keeping the registry alive by itself.
    /// Payloads are dropped by [`Self::maintain`] outside the registry lock.
    pub fn spawn_reaper(self: &Arc<Self>, interval: Duration)
    where
        P: Send + 'static,
    {
        debug_assert!(!interval.is_zero());
        let registry: Weak<Self> = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "credit reaper holds only a weak registry reference and exits after shutdown"
        )]
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let Some(registry) = registry.upgrade() else {
                    break;
                };
                registry.maintain();
            }
        });
    }

    fn maintain_at(&self, now: Instant) -> usize {
        let mut dropped_payloads = Vec::new();
        let changed = {
            let mut inner = self.inner.lock();
            let mut expired = 0;
            let mut purged = 0;
            while inner
                .deadlines
                .first_key_value()
                .is_some_and(|((deadline, _), _)| *deadline <= now)
            {
                let Some(((_deadline, _), (token, kind))) = inner.deadlines.pop_first() else {
                    break;
                };
                match kind {
                    DeadlineKind::Active => {
                        let terminal_until = now + self.terminal_retention;
                        let payload = inner.by_token.get_mut(&token).and_then(|record| {
                            if matches!(record.state, CreditState::Active(_))
                                && now >= record.expires_at
                            {
                                let partition = Arc::clone(&record.binding.partition);
                                transition_terminal(record, CreditState::Expired, terminal_until)
                                    .map(|payload| (payload, partition))
                            } else {
                                None
                            }
                        });
                        if let Some((payload, partition)) = payload {
                            dropped_payloads.push((payload, partition));
                            expired += 1;
                            schedule_deadline(
                                &mut inner,
                                terminal_until,
                                token,
                                DeadlineKind::Terminal,
                            );
                        }
                    }
                    DeadlineKind::Terminal => {
                        let request_id = inner.by_token.get(&token).and_then(|record| {
                            record
                                .terminal_until
                                .filter(|until| now >= *until)
                                .map(|_| Arc::clone(&record.binding.request_id))
                        });
                        if let Some(request_id) = request_id {
                            inner.by_token.remove(&token);
                            inner.by_request.remove(&request_id);
                            purged += 1;
                        }
                    }
                }
            }
            expired + purged
        };
        for (_, partition) in &dropped_payloads {
            sched_metrics::record_capacity_credit_operation(
                partition,
                sched_metrics::capacity_credit_outcome::EXPIRED,
            );
            sched_metrics::decrement_capacity_credit_active(partition);
        }
        drop(dropped_payloads);
        changed
    }

    /// Number of active payload-holding credits.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.inner
            .lock()
            .by_token
            .values()
            .filter(|record| matches!(record.state, CreditState::Active(_)))
            .count()
    }
}

fn record_capacity_credit_error(partition: &str, error: CapacityCreditError) {
    let outcome = match error {
        CapacityCreditError::InvalidToken | CapacityCreditError::InvalidBinding(_) => {
            sched_metrics::capacity_credit_outcome::INVALID_BINDING
        }
        CapacityCreditError::WrongGeneration => {
            sched_metrics::capacity_credit_outcome::WRONG_GENERATION
        }
        CapacityCreditError::RequestConflict => {
            sched_metrics::capacity_credit_outcome::REQUEST_CONFLICT
        }
        CapacityCreditError::Unknown => sched_metrics::capacity_credit_outcome::UNKNOWN,
        CapacityCreditError::BindingMismatch => {
            sched_metrics::capacity_credit_outcome::BINDING_MISMATCH
        }
        CapacityCreditError::Expired => sched_metrics::capacity_credit_outcome::EXPIRED,
        CapacityCreditError::Cancelled => sched_metrics::capacity_credit_outcome::CANCELLED,
        CapacityCreditError::AlreadyRedeemed | CapacityCreditError::AlreadyArmed => {
            sched_metrics::capacity_credit_outcome::REPLAY
        }
        CapacityCreditError::NoCapacity => sched_metrics::capacity_credit_outcome::NO_CAPACITY,
        CapacityCreditError::InternalState => sched_metrics::capacity_credit_outcome::UNKNOWN,
    };
    sched_metrics::record_capacity_credit_operation(partition, outcome);
}

fn schedule_deadline<P>(
    inner: &mut RegistryInner<P>,
    deadline: Instant,
    token: CapacityCreditToken,
    kind: DeadlineKind,
) {
    loop {
        let id = inner.next_deadline_id;
        inner.next_deadline_id = inner.next_deadline_id.wrapping_add(1);
        if let Entry::Vacant(entry) = inner.deadlines.entry((deadline, id)) {
            entry.insert((token, kind));
            return;
        }
    }
}

fn transition_terminal<P>(
    record: &mut CreditRecord<P>,
    terminal: CreditState<P>,
    terminal_until: Instant,
) -> Option<P> {
    if !matches!(record.state, CreditState::Active(_)) {
        return None;
    }
    record.terminal_until = Some(terminal_until);
    match std::mem::replace(&mut record.state, terminal) {
        CreditState::Active(payload) => Some(payload),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    };

    use super::*;

    fn binding(request_id: &str) -> CapacityCreditBinding {
        CapacityCreditBinding::new(
            "green-1",
            7,
            "kimi-k3",
            "kimi-k3",
            TenantKey::new("auth:junu"),
            request_id,
            1_024,
        )
        .unwrap()
    }

    fn registry<P>() -> CapacityCreditRegistry<P> {
        CapacityCreditRegistry::new("green-1", Duration::from_secs(2), Duration::from_secs(3))
            .unwrap()
    }

    #[test]
    fn token_round_trip_is_url_safe_and_debug_redacted() {
        let token = CapacityCreditToken::random();
        let encoded = token.expose_secret();
        assert_eq!(CapacityCreditToken::parse(&encoded).unwrap(), token);
        assert!(!encoded.contains('='));
        assert_eq!(format!("{token:?}"), "CapacityCreditToken([REDACTED])");
        assert_eq!(
            CapacityCreditToken::parse("not a token"),
            Err(CapacityCreditError::InvalidToken)
        );
    }

    #[test]
    fn validates_binding_generation_and_ttls() {
        assert_eq!(
            CapacityCreditBinding::new(" green", 1, "p", "m", TenantKey::new("auth:a"), "r", 1,),
            Err(CapacityCreditError::InvalidBinding("generation"))
        );
        assert!(matches!(
            CapacityCreditRegistry::<()>::new("green", Duration::ZERO, Duration::from_secs(1)),
            Err(CapacityCreditError::InvalidBinding("credit ttl"))
        ));
    }

    #[test]
    fn duplicate_issue_is_idempotent_and_conflict_is_rejected() {
        let registry = registry();
        let first = registry.issue(binding("request-1"), 1).unwrap();
        let repeated = registry.issue(binding("request-1"), 2).unwrap();
        assert!(first.newly_issued);
        assert!(!repeated.newly_issued);
        assert_eq!(first.token, repeated.token);
        assert_eq!(registry.active_len(), 1);

        let mut conflict = binding("request-1");
        conflict.estimated_output_tokens = 2_048;
        assert_eq!(
            registry.issue(conflict, 3),
            Err(CapacityCreditError::RequestConflict)
        );
    }

    #[test]
    fn payload_factory_runs_only_for_a_new_credit() {
        let registry = registry();
        let calls = AtomicUsize::new(0);
        let first = registry
            .issue_with(binding("request-1"), || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok("permit")
            })
            .unwrap();
        let repeated = registry
            .issue_with(binding("request-1"), || {
                calls.fetch_add(1, Ordering::Relaxed);
                Err(CapacityCreditError::NoCapacity)
            })
            .unwrap();
        assert_eq!(first.token, repeated.token);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let denied = registry.issue_with(binding("request-2"), || {
            calls.fetch_add(1, Ordering::Relaxed);
            Err(CapacityCreditError::NoCapacity)
        });
        assert_eq!(denied, Err(CapacityCreditError::NoCapacity));
        assert_eq!(registry.active_len(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn mismatch_does_not_consume_and_redemption_is_single_use() {
        let registry = registry();
        let expected = binding("request-1");
        let issued = registry.issue(expected.clone(), "permit").unwrap();
        let mismatch = binding("request-2");
        assert_eq!(
            registry.redeem(&issued.token, &mismatch).unwrap_err(),
            CapacityCreditError::BindingMismatch
        );
        assert_eq!(registry.active_len(), 1);

        let redeemed = registry.redeem(&issued.token, &expected).unwrap();
        assert_eq!(redeemed.payload, "permit");
        assert_eq!(registry.active_len(), 0);
        assert_eq!(
            registry.redeem(&issued.token, &expected).unwrap_err(),
            CapacityCreditError::AlreadyRedeemed
        );
        assert_eq!(
            registry.issue(expected, "another"),
            Err(CapacityCreditError::AlreadyRedeemed)
        );
    }

    #[test]
    fn concurrent_redemption_has_exactly_one_winner() {
        let registry = Arc::new(registry());
        let expected = binding("request-1");
        let issued = registry.issue(expected.clone(), "permit").unwrap();
        let winners = (0..16)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let token = issued.token.clone();
                let expected = expected.clone();
                std::thread::spawn(move || registry.redeem(&token, &expected).is_ok())
            })
            .map(|thread| usize::from(thread.join().unwrap()))
            .sum::<usize>();
        assert_eq!(winners, 1);
    }

    #[test]
    fn arm_resets_the_deadline_once_and_preserves_the_exact_payload() {
        let registry = registry();
        let start = Instant::now();
        let expected = binding("request-1");
        let issued = registry
            .issue_at(expected.clone(), "held-permit", start)
            .unwrap();
        assert_eq!(issued.expires_at, start + Duration::from_secs(2));

        let armed = registry
            .arm_at(
                &issued.token,
                &expected,
                "dispatch-1",
                start + Duration::from_secs(1),
            )
            .unwrap();
        assert!(armed.newly_armed);
        assert_eq!(armed.expires_at, start + Duration::from_secs(3));

        let repeated = registry
            .arm_at(
                &issued.token,
                &expected,
                "dispatch-1",
                start + Duration::from_millis(1_500),
            )
            .unwrap();
        assert!(!repeated.newly_armed);
        assert_eq!(repeated.expires_at, armed.expires_at);
        assert_eq!(registry.maintain_at(start + Duration::from_secs(2)), 0);

        let redeemed = registry
            .redeem_at(
                &issued.token,
                &expected,
                start + Duration::from_millis(2_500),
            )
            .unwrap();
        assert_eq!(redeemed.payload, "held-permit");
    }

    #[test]
    fn concurrent_arm_calls_have_one_transition_and_one_fixed_deadline() {
        let registry = Arc::new(registry());
        let expected = binding("request-1");
        let issued = registry.issue(expected.clone(), "held-permit").unwrap();
        let results = (0..16)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let token = issued.token.clone();
                let expected = expected.clone();
                std::thread::spawn(move || registry.arm(&token, &expected, "dispatch-1").unwrap())
            })
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            results.iter().filter(|result| result.newly_armed).count(),
            1
        );
        assert!(results
            .iter()
            .all(|result| result.expires_at == results[0].expires_at));
        let redeemed = registry.redeem(&issued.token, &expected).unwrap();
        assert_eq!(redeemed.payload, "held-permit");
    }

    #[test]
    fn concurrent_different_dispatches_have_one_winner_and_conflict_losers() {
        let registry = Arc::new(registry());
        let expected = binding("request-1");
        let issued = registry.issue(expected.clone(), "held-permit").unwrap();
        let results = (0..16)
            .map(|index| {
                let registry = Arc::clone(&registry);
                let token = issued.token.clone();
                let expected = expected.clone();
                std::thread::spawn(move || {
                    registry.arm(&token, &expected, &format!("dispatch-{index}"))
                })
            })
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| { matches!(result, Err(CapacityCreditError::AlreadyArmed)) })
                .count(),
            15
        );
        let redeemed = registry.redeem(&issued.token, &expected).unwrap();
        assert_eq!(redeemed.payload, "held-permit");
    }

    #[test]
    fn arm_fails_closed_for_wrong_binding_and_terminal_states() {
        let registry = registry();
        let start = Instant::now();

        let expected = binding("binding");
        let issued = registry.issue_at(expected.clone(), (), start).unwrap();
        assert_eq!(
            registry.arm_at(
                &CapacityCreditToken::random(),
                &expected,
                "dispatch-1",
                start,
            ),
            Err(CapacityCreditError::Unknown)
        );
        assert_eq!(
            registry.arm_at(&issued.token, &expected, "", start),
            Err(CapacityCreditError::InvalidBinding("dispatch id"))
        );
        assert_eq!(
            registry.arm_at(&issued.token, &expected, &"x".repeat(129), start),
            Err(CapacityCreditError::InvalidBinding("dispatch id"))
        );
        let mut wrong_generation = expected.clone();
        wrong_generation.generation = Arc::from("blue-1");
        assert_eq!(
            registry.arm_at(&issued.token, &wrong_generation, "dispatch-1", start),
            Err(CapacityCreditError::WrongGeneration)
        );
        let mut mismatch = expected.clone();
        mismatch.estimated_output_tokens += 1;
        assert_eq!(
            registry.arm_at(&issued.token, &mismatch, "dispatch-1", start),
            Err(CapacityCreditError::BindingMismatch)
        );
        assert!(registry
            .arm_at(&issued.token, &expected, "dispatch-1", start)
            .is_ok());
        registry.redeem_at(&issued.token, &expected, start).unwrap();

        let cancelled_binding = binding("cancelled");
        let cancelled = registry
            .issue_at(cancelled_binding.clone(), (), start)
            .unwrap();
        assert!(registry.cancel_at(&cancelled.token, start));
        assert_eq!(
            registry.arm_at(
                &cancelled.token,
                &cancelled_binding,
                "dispatch-1",
                start + Duration::from_secs(2),
            ),
            Err(CapacityCreditError::Cancelled)
        );

        let redeemed_binding = binding("redeemed");
        let redeemed = registry
            .issue_at(redeemed_binding.clone(), (), start)
            .unwrap();
        registry
            .redeem_at(&redeemed.token, &redeemed_binding, start)
            .unwrap();
        assert_eq!(
            registry.arm_at(
                &redeemed.token,
                &redeemed_binding,
                "dispatch-1",
                start + Duration::from_secs(2),
            ),
            Err(CapacityCreditError::AlreadyRedeemed)
        );

        let expired_binding = binding("expired");
        let expired = registry
            .issue_at(expired_binding.clone(), (), start)
            .unwrap();
        assert_eq!(
            registry.arm_at(
                &expired.token,
                &expired_binding,
                "dispatch-1",
                start + Duration::from_secs(2),
            ),
            Err(CapacityCreditError::Expired)
        );
        assert_eq!(registry.active_len(), 0);
    }

    struct DropProbe {
        registry: Weak<CapacityCreditRegistry<DropProbe>>,
        dropped_unlocked: Arc<AtomicBool>,
        drop_count: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            if let Some(registry) = self.registry.upgrade() {
                self.dropped_unlocked
                    .store(registry.inner.try_lock().is_some(), Ordering::Release);
            }
            self.drop_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn drop_probe(
        registry: &Arc<CapacityCreditRegistry<DropProbe>>,
        dropped_unlocked: &Arc<AtomicBool>,
        drop_count: &Arc<AtomicUsize>,
    ) -> DropProbe {
        DropProbe {
            registry: Arc::downgrade(registry),
            dropped_unlocked: Arc::clone(dropped_unlocked),
            drop_count: Arc::clone(drop_count),
        }
    }

    #[test]
    fn cancel_and_expiry_drop_payloads_outside_registry_lock() {
        let registry = Arc::new(registry());
        let unlocked = Arc::new(AtomicBool::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        let first = registry
            .issue_at(
                binding("cancel"),
                drop_probe(&registry, &unlocked, &drops),
                start,
            )
            .unwrap();
        assert!(registry.cancel_at(&first.token, start));
        assert!(unlocked.load(Ordering::Acquire));
        assert_eq!(drops.load(Ordering::Relaxed), 1);

        unlocked.store(false, Ordering::Release);
        registry
            .issue_at(
                binding("expire"),
                drop_probe(&registry, &unlocked, &drops),
                start,
            )
            .unwrap();
        assert_eq!(registry.maintain_at(start + Duration::from_secs(2)), 1);
        assert!(unlocked.load(Ordering::Acquire));
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn background_reaper_releases_an_unredeemed_payload() {
        let registry = Arc::new(
            CapacityCreditRegistry::new(
                "green-1",
                Duration::from_millis(20),
                Duration::from_secs(1),
            )
            .unwrap(),
        );
        let unlocked = Arc::new(AtomicBool::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        registry
            .issue(
                binding("expire-in-background"),
                drop_probe(&registry, &unlocked, &drops),
            )
            .unwrap();
        registry.spawn_reaper(Duration::from_millis(5));

        tokio::time::timeout(Duration::from_secs(1), async {
            while drops.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reaper should release the payload after TTL");
        assert!(unlocked.load(Ordering::Acquire));
        assert_eq!(registry.active_len(), 0);
    }

    #[test]
    fn terminal_tombstone_blocks_replay_then_purges() {
        let registry = registry();
        let start = Instant::now();
        let expected = binding("request-1");
        let issued = registry.issue_at(expected.clone(), (), start).unwrap();
        registry
            .redeem_at(&issued.token, &expected, start + Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            registry.issue_at(expected.clone(), (), start + Duration::from_secs(2)),
            Err(CapacityCreditError::AlreadyRedeemed)
        );
        assert_eq!(registry.maintain_at(start + Duration::from_secs(4)), 1);
        assert!(
            registry
                .issue_at(expected, (), start + Duration::from_secs(4))
                .unwrap()
                .newly_issued
        );
    }
}
