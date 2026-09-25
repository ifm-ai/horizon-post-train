//! Scheduler-independent HTTP admission lifecycle accounting.
//!
//! Both the legacy concurrency limiter and the priority scheduler use these
//! guards so `smg_http_admission_*` retains one meaning regardless of which
//! local admission policy is enabled.

use crate::observability::metrics::Metrics;

pub(crate) struct AdmissionActiveGuard {
    outcome_recorded: bool,
}

impl AdmissionActiveGuard {
    pub(crate) fn new() -> Self {
        Metrics::increment_http_admission_active();
        Self {
            outcome_recorded: false,
        }
    }

    pub(crate) fn record_outcome(&mut self, status: u16) {
        if !self.outcome_recorded {
            Metrics::record_http_admission_outcome(status);
            self.outcome_recorded = true;
        }
    }

    pub(crate) fn record_interrupted(&mut self) {
        if !self.outcome_recorded {
            Metrics::record_http_admission_interrupted();
            self.outcome_recorded = true;
        }
    }
}

impl Drop for AdmissionActiveGuard {
    fn drop(&mut self) {
        self.record_interrupted();
        Metrics::decrement_http_admission_active();
    }
}

pub(crate) struct AdmissionPendingGuard {
    resolved: bool,
}

impl AdmissionPendingGuard {
    pub(crate) fn new() -> Self {
        Self { resolved: false }
    }

    pub(crate) fn resolve(&mut self) {
        self.resolved = true;
    }
}

impl Drop for AdmissionPendingGuard {
    fn drop(&mut self) {
        if !self.resolved {
            Metrics::record_http_pre_admission_interrupted();
        }
    }
}

pub(crate) struct AdmissionQueuedGuard;

impl AdmissionQueuedGuard {
    pub(crate) fn new() -> Self {
        Metrics::increment_http_admission_queued();
        Self
    }
}

impl Drop for AdmissionQueuedGuard {
    fn drop(&mut self) {
        Metrics::decrement_http_admission_queued();
    }
}
