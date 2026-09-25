//! Per-class waiter queue used by [`super::engine`] to hold admission
//! candidates while their class is at capacity.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_auth::RequestId;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::{
    engine::SchedulerPermit,
    fair_share::{FairShareCandidate, FairShareProfile, FairShareReservation, GlobalFairShare},
    Class,
};
use crate::tenant::TenantKey;

/// One work-conserving occupancy budget shared by all priority queues.
///
/// Each class keeps its own FIFO lane for dispatch policy, but enqueueing
/// consumes from this common counter. An idle class therefore never strands
/// queue capacity that another class could use.
#[derive(Debug)]
pub struct QueueBudget {
    used: AtomicUsize,
    capacity: usize,
}

impl QueueBudget {
    pub fn new(capacity: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            capacity,
        }
    }

    fn try_acquire(&self) -> bool {
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            if used >= self.capacity {
                return false;
            }
            match self.used.compare_exchange_weak(
                used,
                used + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => used = observed,
            }
        }
    }

    fn release(&self) {
        let released = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_sub(1)
            });
        debug_assert!(released.is_ok(), "queue budget released below zero");
    }

    pub fn depth(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A request waiting for admission in a class queue.
///
/// Carries the class lane to admit under, the wait-start instant (used
/// by the dispatcher's starvation override), the request id (carried
/// so admit can build the inflight handle after dequeue), the
/// cancellation token the client owns (lets the queue GC walkaway
/// clients without admitting them), and the oneshot the dispatcher
/// uses to hand back the [`SchedulerPermit`] when this waiter is
/// admitted. Rejections (queue full, queue timeout, client cancelled)
/// are returned synchronously from `admit` and never go through the
/// channel, so this carries only the success case.
#[derive(Debug)]
pub struct Waiter {
    pub class: Class,
    pub queued_at: Instant,
    pub cancel: CancellationToken,
    pub request_id: RequestId,
    pub permit_tx: oneshot::Sender<SchedulerPermit>,
    pub tenant: Option<TenantKey>,
    pub(crate) fair_share_profile: FairShareProfile,
    pub estimated_output_tokens: u32,
    pub fair_share_reservation: Option<Box<FairShareReservation>>,
}

impl Waiter {
    pub fn new(
        class: Class,
        cancel: CancellationToken,
        request_id: RequestId,
        permit_tx: oneshot::Sender<SchedulerPermit>,
    ) -> Self {
        Self {
            class,
            queued_at: Instant::now(),
            cancel,
            request_id,
            permit_tx,
            tenant: None,
            fair_share_profile: FairShareProfile::Global,
            estimated_output_tokens: 0,
            fair_share_reservation: None,
        }
    }

    pub(crate) fn new_fair(
        class: Class,
        cancel: CancellationToken,
        request_id: RequestId,
        permit_tx: oneshot::Sender<SchedulerPermit>,
        tenant: TenantKey,
        fair_share_profile: FairShareProfile,
        estimated_output_tokens: u32,
    ) -> Self {
        Self {
            class,
            queued_at: Instant::now(),
            cancel,
            request_id,
            permit_tx,
            tenant: Some(tenant),
            fair_share_profile,
            estimated_output_tokens: estimated_output_tokens.max(1),
            fair_share_reservation: None,
        }
    }
}

/// Per-class waiter queue.
///
/// Implemented as a trait so the v1 FIFO impl can be swapped for a
/// per-tenant fair-queue implementation without touching the scheduler.
/// All methods are sync; the queue is a contention point on slot release
/// and admission, so it uses `parking_lot::Mutex` rather than an async
/// mutex.
pub trait ClassQueue: Send + Sync {
    /// Append a waiter. Returns `Err(waiter)` when the shared global queue
    /// budget is exhausted so the caller can convert the rejection into a
    /// saturation response.
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter>;

    /// Pop the next waiter, or `None` when empty.
    fn pop_eligible(&self) -> Option<Waiter>;

    /// Wall-clock age of the head waiter (used by the dispatcher's
    /// starvation override). `None` when empty.
    fn head_age(&self) -> Option<Duration>;

    /// Current depth, including waiters whose cancel token has fired.
    fn depth(&self) -> usize;

    /// Configured soft share of the global queue budget, for metrics.
    fn capacity(&self) -> usize;

    /// Drain any leading run of waiters whose cancel token has fired
    /// (clients that walked away while queued). One call removes every
    /// consecutive cancelled head in a single lock acquisition; the
    /// first non-cancelled or empty position stops the scan.
    fn drop_cancelled_head(&self);
}

/// First-in, first-out per-class queue backed by a shared occupancy budget.
pub struct FifoClassQueue {
    waiters: Mutex<VecDeque<Waiter>>,
    soft_limit: usize,
    budget: Arc<QueueBudget>,
}

impl FifoClassQueue {
    /// Construct a standalone queue. Production scheduler queues use
    /// [`Self::with_shared_budget`]; the private budget keeps the queue useful
    /// in focused tests and for future isolated callers.
    pub fn new(max: usize) -> Self {
        Self::with_shared_budget(max, Arc::new(QueueBudget::new(max)))
    }

    pub fn with_shared_budget(soft_limit: usize, budget: Arc<QueueBudget>) -> Self {
        Self {
            waiters: Mutex::new(VecDeque::with_capacity(soft_limit.min(64))),
            soft_limit,
            budget,
        }
    }
}

impl ClassQueue for FifoClassQueue {
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter> {
        if !self.budget.try_acquire() {
            return Err(waiter);
        }
        let mut guard = self.waiters.lock();
        guard.push_back(waiter);
        Ok(())
    }

    fn pop_eligible(&self) -> Option<Waiter> {
        let waiter = self.waiters.lock().pop_front();
        if waiter.is_some() {
            self.budget.release();
        }
        waiter
    }

    fn head_age(&self) -> Option<Duration> {
        self.waiters.lock().front().map(|w| w.queued_at.elapsed())
    }

    fn depth(&self) -> usize {
        self.waiters.lock().len()
    }

    fn capacity(&self) -> usize {
        self.soft_limit
    }

    fn drop_cancelled_head(&self) {
        let mut guard = self.waiters.lock();
        while guard.front().is_some_and(|w| w.cancel.is_cancelled()) {
            guard.pop_front();
            self.budget.release();
        }
    }
}

/// Work-conserving per-partition queue ordered by one shared token ledger.
///
/// Only waiters in this concrete queue are candidates, so debt in another
/// non-fungible model pool can never idle local capacity.
pub struct FairClassQueue {
    waiters: Mutex<VecDeque<Waiter>>,
    soft_limit: usize,
    budget: Arc<QueueBudget>,
    ledger: Arc<GlobalFairShare>,
    scope_id: u64,
    class: Class,
}

impl FairClassQueue {
    pub fn with_shared_budget(
        class: Class,
        soft_limit: usize,
        budget: Arc<QueueBudget>,
        ledger: Arc<GlobalFairShare>,
        scope_id: u64,
    ) -> Self {
        Self {
            waiters: Mutex::new(VecDeque::with_capacity(soft_limit.min(64))),
            soft_limit,
            budget,
            ledger,
            scope_id,
            class,
        }
    }
}

impl ClassQueue for FairClassQueue {
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter> {
        let Some(tenant) = waiter.tenant.as_ref() else {
            return Err(waiter);
        };
        if !self.budget.try_acquire() {
            return Err(waiter);
        }
        let mut guard = self.waiters.lock();
        self.ledger.register_waiter_in_profile(
            self.scope_id,
            &waiter.fair_share_profile,
            tenant,
            self.class,
        );
        guard.push_back(waiter);
        Ok(())
    }

    fn pop_eligible(&self) -> Option<Waiter> {
        let mut guard = self.waiters.lock();
        let profile = guard
            .iter()
            .filter(|waiter| !waiter.cancel.is_cancelled())
            .min_by_key(|waiter| waiter.queued_at)
            .map(|waiter| waiter.fair_share_profile.clone())?;
        let selection = {
            let candidates: Vec<_> = guard
                .iter()
                .enumerate()
                .filter(|(_, waiter)| !waiter.cancel.is_cancelled())
                .filter(|(_, waiter)| waiter.fair_share_profile == profile)
                .filter_map(|(index, waiter)| {
                    waiter.tenant.as_ref().map(|tenant| FairShareCandidate {
                        index,
                        tenant,
                        estimated_output_tokens: waiter.estimated_output_tokens,
                    })
                })
                .collect();
            self.ledger.reserve_local_candidate_in_profile(
                self.scope_id,
                &profile,
                self.class,
                &candidates,
            )
        }?;
        let Some(mut waiter) = guard.remove(selection.index) else {
            selection.reservation.cancel();
            return None;
        };
        self.budget.release();
        if let Some(tenant) = waiter.tenant.as_ref() {
            self.ledger.record_queue_wait(
                &waiter.fair_share_profile,
                tenant,
                self.class,
                waiter.queued_at.elapsed(),
            );
        }
        waiter.fair_share_reservation = Some(Box::new(selection.reservation));
        Some(waiter)
    }

    fn head_age(&self) -> Option<Duration> {
        self.waiters
            .lock()
            .iter()
            .map(|waiter| waiter.queued_at.elapsed())
            .max()
    }

    fn depth(&self) -> usize {
        self.waiters.lock().len()
    }

    fn capacity(&self) -> usize {
        self.soft_limit
    }

    fn drop_cancelled_head(&self) {
        let mut guard = self.waiters.lock();
        let mut index = 0;
        while index < guard.len() {
            if !guard[index].cancel.is_cancelled() {
                index += 1;
                continue;
            }
            let Some(waiter) = guard.remove(index) else {
                break;
            };
            self.budget.release();
            if let Some(tenant) = waiter.tenant.as_ref() {
                self.ledger.remove_waiter_in_profile(
                    self.scope_id,
                    &waiter.fair_share_profile,
                    tenant,
                    self.class,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::middleware::scheduler::{Class, FairShareConfig, ModelFairShareConfig};

    fn waiter(class: Class) -> Waiter {
        let (tx, _rx) = oneshot::channel();
        Waiter::new(class, CancellationToken::new(), RequestId("t".into()), tx)
    }

    fn waiter_with_cancel(class: Class, cancel: CancellationToken) -> Waiter {
        let (tx, _rx) = oneshot::channel();
        Waiter::new(class, cancel, RequestId("t".into()), tx)
    }

    fn model_ledger() -> Arc<GlobalFairShare> {
        Arc::new(GlobalFairShare::from_config(&FairShareConfig {
            default_weight: 1.0,
            default_output_tokens: 10,
            trust_output_token_estimate_header: false,
            trust_request_model_header: true,
            tenant_weights: HashMap::new(),
            model_profiles: HashMap::from([
                (
                    "model-a".to_string(),
                    ModelFairShareConfig {
                        tenant_weights: HashMap::new(),
                        other_weight: 1.0,
                    },
                ),
                (
                    "model-b".to_string(),
                    ModelFairShareConfig {
                        tenant_weights: HashMap::new(),
                        other_weight: 1.0,
                    },
                ),
            ]),
        }))
    }

    fn model_waiter(id: &str, tenant: &str, profile: FairShareProfile) -> Waiter {
        let (tx, _rx) = oneshot::channel();
        Waiter::new_fair(
            Class::Default,
            CancellationToken::new(),
            RequestId(id.into()),
            tx,
            TenantKey::new(tenant),
            profile,
            10,
        )
    }

    #[test]
    fn test_try_enqueue_fills_to_capacity_then_rejects() {
        let q = FifoClassQueue::new(4);
        for _ in 0..4 {
            assert!(q.try_enqueue(waiter(Class::Default)).is_ok());
        }
        let rejected = q
            .try_enqueue(waiter(Class::Default))
            .expect_err("queue full");
        assert_eq!(rejected.class, Class::Default);
    }

    #[test]
    fn test_pop_eligible_is_fifo() {
        // Distinguish waiters by class to verify insertion order is preserved.
        let q = FifoClassQueue::new(4);
        q.try_enqueue(waiter(Class::Bulk)).unwrap();
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.pop_eligible().unwrap().class, Class::Bulk);
        assert_eq!(q.pop_eligible().unwrap().class, Class::Default);
        assert_eq!(q.pop_eligible().unwrap().class, Class::Interactive);
        assert!(q.pop_eligible().is_none());
    }

    #[test]
    fn test_pop_eligible_returns_none_when_empty() {
        let q = FifoClassQueue::new(4);
        assert!(q.pop_eligible().is_none());
    }

    #[test]
    fn test_head_age_is_some_after_enqueue() {
        let q = FifoClassQueue::new(4);
        assert!(q.head_age().is_none());
        q.try_enqueue(waiter(Class::Default)).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let age = q.head_age().expect("head present");
        assert!(age >= Duration::from_millis(5));
    }

    #[test]
    fn test_depth_reflects_enqueue_and_pop() {
        let q = FifoClassQueue::new(4);
        assert_eq!(q.depth(), 0);
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.try_enqueue(waiter(Class::Default)).unwrap();
        assert_eq!(q.depth(), 2);
        q.pop_eligible();
        assert_eq!(q.depth(), 1);
    }

    #[test]
    fn test_drop_cancelled_head_removes_cancelled_head_only() {
        let q = FifoClassQueue::new(4);
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
            .unwrap();
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.depth(), 2);

        q.drop_cancelled_head();
        assert_eq!(q.depth(), 1, "cancelled head dropped");
        assert_eq!(
            q.pop_eligible().unwrap().class,
            Class::Interactive,
            "live waiter exposed as new head"
        );
    }

    #[test]
    fn oldest_model_group_wins_before_model_local_virtual_service() {
        let ledger = model_ledger();
        let scope_id = ledger.new_scope();
        let queue = FairClassQueue::with_shared_budget(
            Class::Default,
            8,
            Arc::new(QueueBudget::new(8)),
            Arc::clone(&ledger),
            scope_id,
        );
        let mut older = model_waiter(
            "older-model-a",
            "header:a",
            ledger.profile_for_model("model-a"),
        );
        older.queued_at = Instant::now() - Duration::from_secs(1);
        let newer = model_waiter(
            "newer-model-b",
            "header:b",
            ledger.profile_for_model("model-b"),
        );
        queue.try_enqueue(older).unwrap();
        queue.try_enqueue(newer).unwrap();

        let mut first = queue.pop_eligible().expect("oldest model group dispatches");
        assert_eq!(first.request_id.0, "older-model-a");
        first
            .fair_share_reservation
            .take()
            .expect("model waiter has reservation")
            .cancel();
        let mut second = queue.pop_eligible().expect("other model remains eligible");
        assert_eq!(second.request_id.0, "newer-model-b");
        second
            .fair_share_reservation
            .take()
            .expect("model waiter has reservation")
            .cancel();
    }

    #[test]
    fn test_drop_cancelled_head_leaves_live_head_alone() {
        let q = FifoClassQueue::new(4);
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.drop_cancelled_head();
        assert_eq!(q.depth(), 1, "non-cancelled head retained");
    }

    #[test]
    fn test_drop_cancelled_head_noop_on_empty_queue() {
        let q = FifoClassQueue::new(4);
        q.drop_cancelled_head();
        assert_eq!(q.depth(), 0);
    }

    #[test]
    fn test_drop_cancelled_head_drains_run_in_single_call() {
        // When several clients in a row have walked away, a single call
        // should clear the whole run rather than requiring the caller to
        // loop. This keeps the dispatcher's GC pass to one lock cycle
        // instead of N.
        let q = FifoClassQueue::new(8);
        for _ in 0..3 {
            let cancelled = CancellationToken::new();
            cancelled.cancel();
            q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
                .unwrap();
        }
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.depth(), 4);

        q.drop_cancelled_head();
        assert_eq!(
            q.depth(),
            1,
            "all three cancelled heads dropped in one call"
        );
        assert_eq!(
            q.pop_eligible().unwrap().class,
            Class::Interactive,
            "first live waiter is now the head"
        );
    }

    #[test]
    fn test_shared_budget_is_work_conserving_across_class_queues() {
        let budget = Arc::new(QueueBudget::new(3));
        let default = FifoClassQueue::with_shared_budget(1, Arc::clone(&budget));
        let bulk = FifoClassQueue::with_shared_budget(2, Arc::clone(&budget));

        default.try_enqueue(waiter(Class::Default)).unwrap();
        default.try_enqueue(waiter(Class::Default)).unwrap();
        default.try_enqueue(waiter(Class::Default)).unwrap();
        assert_eq!(default.depth(), 3, "default borrows both idle shares");
        assert_eq!(default.capacity(), 1, "configured share remains visible");
        assert_eq!(budget.depth(), 3);
        assert!(bulk.try_enqueue(waiter(Class::Bulk)).is_err());

        default.pop_eligible().unwrap();
        assert_eq!(budget.depth(), 2);
        bulk.try_enqueue(waiter(Class::Bulk)).unwrap();
        assert_eq!(budget.depth(), 3);
    }

    #[test]
    fn test_cancelled_waiter_releases_shared_budget() {
        let budget = Arc::new(QueueBudget::new(1));
        let default = FifoClassQueue::with_shared_budget(1, Arc::clone(&budget));
        let bulk = FifoClassQueue::with_shared_budget(0, Arc::clone(&budget));
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        default
            .try_enqueue(waiter_with_cancel(Class::Default, cancelled))
            .unwrap();
        assert!(bulk.try_enqueue(waiter(Class::Bulk)).is_err());

        default.drop_cancelled_head();
        assert_eq!(budget.depth(), 0);
        bulk.try_enqueue(waiter(Class::Bulk)).unwrap();
    }

    #[test]
    fn test_shared_budget_never_overadmits_under_concurrency() {
        let budget = Arc::new(QueueBudget::new(64));
        let queue = Arc::new(FifoClassQueue::with_shared_budget(16, budget));
        let threads: Vec<_> = (0..200)
            .map(|_| {
                let queue = Arc::clone(&queue);
                std::thread::spawn(move || queue.try_enqueue(waiter(Class::Default)).is_ok())
            })
            .collect();
        let admitted = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|admitted| *admitted)
            .count();
        assert_eq!(admitted, 64);
        assert_eq!(queue.depth(), 64);
    }
}
