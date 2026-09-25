//! Prometheus metrics for the priority scheduler.
//!
//! Split to match design §9: this module holds the **operational** counters
//! and the queue-wait histogram (recorded on the admission / queue paths),
//! plus the **capacity / autoscaling** gauge setters. The gauges are
//! point-in-time state (inflight, queue depth, utilization, capacity
//! pressure) refreshed by the scheduler's sampler task, so the hot
//! admission path only does cheap counter increments.
//!
//! All names carry the `smg_` prefix to match the rest of the gateway's
//! metrics. Class/outcome labels are `&'static str` (no per-request
//! allocation).

use std::time::Duration;

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

use super::Class;
use crate::observability::metrics::intern_string;

const ADMIT_TOTAL: &str = "smg_scheduler_admit_total";
const QUEUE_WAIT_SECONDS: &str = "smg_scheduler_queue_wait_seconds";
const PREEMPTION_TOTAL: &str = "smg_scheduler_preemption_total";
const CLAMP_TOTAL: &str = "smg_scheduler_clamp_total";
const UNKNOWN_PRIORITY_TOTAL: &str = "smg_scheduler_unknown_priority_value_total";
const STARVATION_PROMOTION_TOTAL: &str = "smg_scheduler_starvation_promotion_total";
const PARTITION_ADMIT_TOTAL: &str = "smg_scheduler_partition_admit_total";
const FAIR_SHARE_CHARGED_OUTPUT_TOKENS_TOTAL: &str = "smg_fair_share_charged_output_tokens_total";
const FAIR_SHARE_VIRTUAL_FINISH: &str = "smg_fair_share_virtual_finish";
const FAIR_SHARE_OTHER_BUCKET_VIRTUAL_FINISH: &str = "smg_fair_share_other_bucket_virtual_finish";
const FAIR_SHARE_RESERVED_OUTPUT_TOKENS: &str = "smg_fair_share_reserved_output_tokens";
const FAIR_SHARE_QUEUE_WAIT_SECONDS: &str = "smg_fair_share_queue_wait_seconds";
const FAIR_SHARE_FALLBACK_TOTAL: &str = "smg_fair_share_fallback_total";
const FAIR_SHARE_UNKNOWN_TENANT_TOTAL: &str = "smg_fair_share_unknown_tenant_total";
const CAPACITY_CREDIT_OPERATIONS_TOTAL: &str = "smg_capacity_credit_operations_total";
const CAPACITY_CREDIT_ACTIVE: &str = "smg_capacity_credit_active";

// Capacity / autoscaling gauges, refreshed by the sampler task.
const INFLIGHT: &str = "smg_scheduler_inflight";
const QUEUE_DEPTH: &str = "smg_scheduler_queue_depth";
const UTILIZATION: &str = "smg_scheduler_utilization";
const QUEUE_SIZE_LIMIT: &str = "smg_scheduler_queue_size_limit";
const RETRY_AFTER_SECONDS: &str = "smg_scheduler_retry_after_seconds";
const CLASS_CAPACITY_PRESSURE: &str = "smg_scheduler_class_capacity_pressure";
const PARTITION_CAPACITY: &str = "smg_scheduler_partition_capacity";
const PARTITION_HEALTHY_REPLICAS: &str = "smg_scheduler_partition_healthy_replicas";
const PARTITION_INFLIGHT: &str = "smg_scheduler_partition_inflight";
const PARTITION_QUEUE_DEPTH: &str = "smg_scheduler_partition_queue_depth";
const PARTITION_QUEUE_SIZE_LIMIT: &str = "smg_scheduler_partition_queue_size_limit";
const PARTITION_UTILIZATION: &str = "smg_scheduler_partition_utilization";

/// `outcome` label values for [`record_admit`].
pub mod outcome {
    /// Admitted (fast path or after queueing — not distinguished).
    pub const ADMITTED: &str = "admitted";
    /// Shared global queue budget was exhausted.
    pub const REJECTED_QUEUE_FULL: &str = "rejected_queue_full";
    /// Queued waiter aged past `queue_timeout`.
    pub const REJECTED_QUEUE_TIMEOUT: &str = "rejected_queue_timeout";
    /// The request was admitted but then preempted before producing a byte.
    pub const PREEMPTED: &str = "preempted";
    /// The caller's client disconnected before admission completed.
    pub const CLIENT_CANCELLED: &str = "client_cancelled";
}

/// Bounded `outcome` values for capacity-credit protocol metrics.
pub mod capacity_credit_outcome {
    pub const ISSUED: &str = "issued";
    pub const IDEMPOTENT_RETRY: &str = "idempotent_retry";
    pub const ARMED: &str = "armed";
    pub const ARM_IDEMPOTENT_RETRY: &str = "arm_idempotent_retry";
    pub const ARM_REJECTED: &str = "arm_rejected";
    pub const REDEEMED: &str = "redeemed";
    pub const CANCELLED: &str = "cancelled";
    pub const EXPIRED: &str = "expired";
    pub const NO_CAPACITY: &str = "no_capacity";
    pub const INVALID_BINDING: &str = "invalid_binding";
    pub const BINDING_MISMATCH: &str = "binding_mismatch";
    pub const REQUEST_CONFLICT: &str = "request_conflict";
    pub const WRONG_GENERATION: &str = "wrong_generation";
    pub const REPLAY: &str = "replay";
    pub const UNKNOWN: &str = "unknown";
}

/// Register descriptions. Called once from `observability::metrics::init_metrics`.
pub fn describe() {
    describe_counter!(
        ADMIT_TOTAL,
        "Priority-scheduler admission outcomes by class and outcome"
    );
    describe_histogram!(
        QUEUE_WAIT_SECONDS,
        "Time a request spent queued before admission, timeout, or cancel"
    );
    describe_counter!(
        PREEMPTION_TOTAL,
        "Successful preemptions by victim class and preempting class"
    );
    describe_counter!(
        CLAMP_TOTAL,
        "Requests whose priority was clamped below the requested class by tenant policy"
    );
    describe_counter!(
        UNKNOWN_PRIORITY_TOTAL,
        "Requests with an unrecognized priority header value (treated as default)"
    );
    describe_counter!(
        STARVATION_PROMOTION_TOTAL,
        "Queued waiters admitted via the starvation override path"
    );
    describe_counter!(
        PARTITION_ADMIT_TOTAL,
        "Priority-scheduler admission outcomes by partition, class, and outcome"
    );
    describe_counter!(
        FAIR_SHARE_CHARGED_OUTPUT_TOKENS_TOTAL,
        "Output tokens charged to the global fair-share ledger by tenant"
    );
    describe_gauge!(
        FAIR_SHARE_VIRTUAL_FINISH,
        "Active-set normalized tenant virtual finish by model profile"
    );
    describe_gauge!(
        FAIR_SHARE_OTHER_BUCKET_VIRTUAL_FINISH,
        "Outer aggregate other-bucket virtual finish by model profile"
    );
    describe_gauge!(
        FAIR_SHARE_RESERVED_OUTPUT_TOKENS,
        "Provisional output-token charges for active requests by tenant"
    );
    describe_histogram!(
        FAIR_SHARE_QUEUE_WAIT_SECONDS,
        "Fair-share queue wait by tenant and priority class"
    );
    describe_counter!(
        FAIR_SHARE_FALLBACK_TOTAL,
        "Fair-share settlements that used an estimate instead of terminal usage"
    );
    describe_counter!(
        FAIR_SHARE_UNKNOWN_TENANT_TOTAL,
        "Requests whose resolved tenant has no explicit fair-share weight"
    );
    describe_counter!(
        CAPACITY_CREDIT_OPERATIONS_TOTAL,
        "Capacity-credit lifecycle and protocol outcomes by configured admission partition"
    );
    describe_gauge!(
        CAPACITY_CREDIT_ACTIVE,
        "Active unredeemed capacity credits holding scheduler slots by admission partition"
    );
    describe_gauge!(INFLIGHT, "Current in-flight request count per class");
    describe_gauge!(QUEUE_DEPTH, "Current queued waiter count per class");
    describe_gauge!(
        UTILIZATION,
        "Total in-flight requests divided by backend capacity (0.0-1.0+)"
    );
    describe_gauge!(
        QUEUE_SIZE_LIMIT,
        "Configured soft share of the work-conserving global queue per class"
    );
    describe_gauge!(
        RETRY_AFTER_SECONDS,
        "Current Retry-After estimate in whole seconds by priority class"
    );
    describe_gauge!(
        CLASS_CAPACITY_PRESSURE,
        "Normalized 0.0-1.0 per-class pressure (max of queue and slot pressure)"
    );
    describe_gauge!(
        PARTITION_CAPACITY,
        "Current hard admission capacity by partition"
    );
    describe_gauge!(
        PARTITION_HEALTHY_REPLICAS,
        "Current healthy worker count used to weight each admission partition"
    );
    describe_gauge!(
        PARTITION_INFLIGHT,
        "Current in-flight request count by admission partition and class"
    );
    describe_gauge!(
        PARTITION_QUEUE_DEPTH,
        "Current queued waiter count by admission partition and class"
    );
    describe_gauge!(
        PARTITION_QUEUE_SIZE_LIMIT,
        "Configured queue budget share by admission partition and class"
    );
    describe_gauge!(
        PARTITION_UTILIZATION,
        "Partition in-flight requests divided by partition capacity"
    );
}

/// Record the outcome of an admission attempt for `class`.
pub fn record_admit(class: Class, outcome: &'static str) {
    counter!(ADMIT_TOTAL, "class" => class.as_str(), "outcome" => outcome).increment(1);
}

pub fn record_partition_admit(partition: &str, class: Class, outcome: &'static str) {
    counter!(
        PARTITION_ADMIT_TOTAL,
        "partition" => intern_string(partition),
        "class" => class.as_str(),
        "outcome" => outcome
    )
    .increment(1);
}

/// Record the time a request waited in a class queue.
pub fn record_queue_wait(class: Class, wait: Duration) {
    histogram!(QUEUE_WAIT_SECONDS, "class" => class.as_str()).record(wait.as_secs_f64());
}

/// Record a successful preemption.
pub fn record_preemption(victim_class: Class, by_class: Class) {
    counter!(
        PREEMPTION_TOTAL,
        "victim_class" => victim_class.as_str(),
        "by_class" => by_class.as_str()
    )
    .increment(1);
}

/// Record a priority clamp (only when the effective class is below the
/// requested class). `tenant` is interned — clamps are rare, so its
/// cardinality is bounded by the set of tenants that actually over-ask.
pub fn record_clamp(requested: Class, effective: Class, tenant: &str) {
    counter!(
        CLAMP_TOTAL,
        "tenant" => intern_string(tenant),
        "requested_class" => requested.as_str(),
        "effective_class" => effective.as_str()
    )
    .increment(1);
}

/// Record an unrecognized priority header value. `tenant` is interned (bad
/// header values are rare), and is the actionable dimension — it tells ops
/// which tenant is mis-setting the priority header.
pub fn record_unknown_priority(tenant: &str) {
    counter!(UNKNOWN_PRIORITY_TOTAL, "tenant" => intern_string(tenant)).increment(1);
}

/// Record a starvation-override promotion.
pub fn record_starvation_promotion(class: Class) {
    counter!(STARVATION_PROMOTION_TOTAL, "class" => class.as_str()).increment(1);
}

pub fn record_fair_share_charged_output_tokens(tenant: &str, tokens: u32) {
    counter!(
        FAIR_SHARE_CHARGED_OUTPUT_TOKENS_TOTAL,
        "tenant" => intern_string(tenant)
    )
    .increment(u64::from(tokens));
}

pub fn set_fair_share_virtual_finish(model: &str, tenant: &str, virtual_finish: f64) {
    gauge!(
        FAIR_SHARE_VIRTUAL_FINISH,
        "model" => intern_string(model),
        "tenant" => intern_string(tenant)
    )
    .set(virtual_finish);
}

pub fn set_fair_share_other_bucket_virtual_finish(model: &str, virtual_finish: f64) {
    gauge!(
        FAIR_SHARE_OTHER_BUCKET_VIRTUAL_FINISH,
        "model" => intern_string(model)
    )
    .set(virtual_finish);
}

pub fn set_fair_share_reserved_output_tokens(tenant: &str, tokens: u64) {
    gauge!(
        FAIR_SHARE_RESERVED_OUTPUT_TOKENS,
        "tenant" => intern_string(tenant)
    )
    .set(tokens as f64);
}

pub fn record_fair_share_queue_wait(model: &str, tenant: &str, class: Class, wait: Duration) {
    histogram!(
        FAIR_SHARE_QUEUE_WAIT_SECONDS,
        "model" => intern_string(model),
        "tenant" => intern_string(tenant),
        "class" => class.as_str()
    )
    .record(wait.as_secs_f64());
}

pub fn record_fair_share_fallback(reason: &'static str) {
    counter!(FAIR_SHARE_FALLBACK_TOTAL, "reason" => reason).increment(1);
}

pub fn record_fair_share_unknown_tenant(tenant: &str) {
    counter!(
        FAIR_SHARE_UNKNOWN_TENANT_TOTAL,
        "tenant" => intern_string(tenant)
    )
    .increment(1);
}

pub fn record_capacity_credit_operation(partition: &str, outcome: &'static str) {
    counter!(
        CAPACITY_CREDIT_OPERATIONS_TOTAL,
        "partition" => intern_string(partition),
        "outcome" => outcome
    )
    .increment(1);
}

pub fn increment_capacity_credit_active(partition: &str) {
    gauge!(
        CAPACITY_CREDIT_ACTIVE,
        "partition" => intern_string(partition)
    )
    .increment(1.0);
}

pub fn decrement_capacity_credit_active(partition: &str) {
    gauge!(
        CAPACITY_CREDIT_ACTIVE,
        "partition" => intern_string(partition)
    )
    .decrement(1.0);
}

/// Set the in-flight gauge for a class (sampler).
pub fn set_inflight(class: Class, count: u16) {
    gauge!(INFLIGHT, "class" => class.as_str()).set(f64::from(count));
}

/// Set the queue-depth gauge for a class (sampler).
pub fn set_queue_depth(class: Class, depth: usize) {
    gauge!(QUEUE_DEPTH, "class" => class.as_str()).set(depth as f64);
}

/// Set the overall utilization gauge: total in-flight / capacity (sampler).
pub fn set_utilization(utilization: f64) {
    gauge!(UTILIZATION).set(utilization);
}

/// Set the queue-size-limit gauge for a class (sampler).
pub fn set_queue_size_limit(class: Class, limit: usize) {
    gauge!(QUEUE_SIZE_LIMIT, "class" => class.as_str()).set(limit as f64);
}

/// Set the current Retry-After estimate for a class (sampler).
pub fn set_retry_after_seconds(class: Class, seconds: u64) {
    gauge!(RETRY_AFTER_SECONDS, "class" => class.as_str()).set(seconds as f64);
}

/// Set the normalized capacity-pressure gauge for a class (sampler).
pub fn set_class_capacity_pressure(class: Class, pressure: f64) {
    gauge!(CLASS_CAPACITY_PRESSURE, "class" => class.as_str()).set(pressure);
}

pub fn set_partition_capacity(partition: &str, capacity: u16) {
    gauge!(PARTITION_CAPACITY, "partition" => intern_string(partition)).set(f64::from(capacity));
}

pub fn set_partition_healthy_replicas(partition: &str, replicas: u16) {
    gauge!(PARTITION_HEALTHY_REPLICAS, "partition" => intern_string(partition))
        .set(f64::from(replicas));
}

pub fn set_partition_inflight(partition: &str, class: Class, count: u16) {
    gauge!(
        PARTITION_INFLIGHT,
        "partition" => intern_string(partition),
        "class" => class.as_str()
    )
    .set(f64::from(count));
}

pub fn set_partition_queue_depth(partition: &str, class: Class, depth: usize) {
    gauge!(
        PARTITION_QUEUE_DEPTH,
        "partition" => intern_string(partition),
        "class" => class.as_str()
    )
    .set(depth as f64);
}

pub fn set_partition_queue_size_limit(partition: &str, class: Class, limit: usize) {
    gauge!(
        PARTITION_QUEUE_SIZE_LIMIT,
        "partition" => intern_string(partition),
        "class" => class.as_str()
    )
    .set(limit as f64);
}

pub fn set_partition_utilization(partition: &str, utilization: f64) {
    gauge!(PARTITION_UTILIZATION, "partition" => intern_string(partition)).set(utilization);
}
