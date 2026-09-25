//! Unified worker activation step.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::info;
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    worker::{Worker, WorkerRegistry},
    workflow::data::WorkerRegistrationData,
};

fn activation_failed(worker_url: &str) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new("activate_workers"),
        message: format!("Worker {worker_url} is no longer the worker registered by this workflow"),
    }
}

/// Promote the exact registered worker objects through the registry so every
/// lifecycle subscriber observes the transition. In particular,
/// `WorkerCapacity` and the partition-capacity coordinator both derive their
/// snapshots from `WorkerEvent`; mutating a worker directly leaves those
/// snapshots stale until an unrelated registry event arrives.
fn activate_registered_workers(
    registry: &WorkerRegistry,
    workers: &[Arc<dyn Worker>],
) -> WorkflowResult<()> {
    for worker in workers {
        let worker_id = registry
            .get_id_by_url(worker.url())
            .ok_or_else(|| activation_failed(worker.url()))?;
        let expected_revision = worker.revision();

        // URL resolution alone is not enough: another workflow may have
        // replaced or recreated the worker at this URL. Check Arc identity and
        // transition under the same per-worker mutation lock so stale workflow
        // completion cannot activate the replacement.
        let outcome = registry.apply_if_revision(&worker_id, expected_revision, |current| {
            let exact_worker = Arc::ptr_eq(current, worker);
            (exact_worker, exact_worker.then_some(WorkerStatus::Ready))
        });

        if !matches!(outcome, Some((true, _))) {
            return Err(activation_failed(worker.url()));
        }
    }

    Ok(())
}

/// Final step in any worker registration workflow: flip Pending → Ready.
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        activate_registered_workers(&app_context.worker_registry, workers)?;

        info!("Activated {} worker(s)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use tokio::{sync::broadcast::error::TryRecvError, time::timeout};

    use super::*;
    use crate::worker::{
        event::WorkerEvent, BasicWorkerBuilder, CapacitySource, CapacityTrackerSettings,
        WorkerCapacity,
    };

    fn worker_with_capacity(url: &str, capacity: u16, status: WorkerStatus) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert("max_running_requests".to_string(), capacity.to_string());
        Arc::new(
            BasicWorkerBuilder::new(url)
                .labels(labels)
                .status(status)
                .build(),
        )
    }

    async fn wait_for_capacity(rx: &mut tokio::sync::watch::Receiver<u16>, expected: u16) {
        timeout(Duration::from_secs(2), async {
            loop {
                if *rx.borrow_and_update() == expected {
                    break;
                }
                rx.changed().await.expect("capacity watch closed");
            }
        })
        .await
        .unwrap_or_else(|_| panic!("worker capacity never reached {expected}"));
    }

    #[test]
    fn activation_emits_status_changed_for_pending_workers_only() {
        let registry = WorkerRegistry::new();
        let pending = worker_with_capacity("http://pending:8000", 38, WorkerStatus::Pending);
        let ready = worker_with_capacity("http://ready:8000", 38, WorkerStatus::Ready);
        let pending_id = registry.register(Arc::clone(&pending)).unwrap();
        registry.register(Arc::clone(&ready)).unwrap();

        // Subscribe after registration so only activation events are visible.
        let mut events = registry.subscribe_events();
        activate_registered_workers(&registry, &[pending, ready]).unwrap();

        match events.try_recv().expect("pending worker status event") {
            WorkerEvent::StatusChanged {
                worker_id,
                old_status,
                new_status,
                ..
            } => {
                assert_eq!(worker_id, pending_id);
                assert_eq!(old_status, WorkerStatus::Pending);
                assert_eq!(new_status, WorkerStatus::Ready);
            }
            event => panic!("unexpected activation event: {event:?}"),
        }
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn stale_activation_cannot_activate_same_url_replacement() {
        let registry = WorkerRegistry::new();
        let original = worker_with_capacity("http://same:8000", 38, WorkerStatus::Pending);
        let worker_id = registry.register(Arc::clone(&original)).unwrap();
        let replacement = worker_with_capacity("http://same:8000", 38, WorkerStatus::Pending);
        assert!(registry.replace(&worker_id, Arc::clone(&replacement)));
        assert!(Arc::ptr_eq(
            &registry.get(&worker_id).unwrap(),
            &replacement
        ));

        // Subscribe after replacement so any event can only come from stale activation.
        let mut events = registry.subscribe_events();
        assert!(activate_registered_workers(&registry, &[original]).is_err());
        assert_eq!(replacement.status(), WorkerStatus::Pending);
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn activation_recomputes_aggregate_worker_capacity() {
        let registry = Arc::new(WorkerRegistry::new());
        let sentinel = worker_with_capacity("http://sentinel:8000", 1, WorkerStatus::Ready);
        let sentinel_id = registry.register(sentinel).unwrap();
        let tracker = WorkerCapacity::spawn(
            Arc::clone(&registry),
            CapacityTrackerSettings {
                legacy_max_concurrent_requests: 512,
                ..CapacityTrackerSettings::default()
            },
        );
        let mut capacity = tracker.watch();
        assert_eq!(tracker.current(), 1);

        let first = worker_with_capacity("http://first:8000", 38, WorkerStatus::Pending);
        let second = worker_with_capacity("http://second:8000", 38, WorkerStatus::Pending);
        registry.register(Arc::clone(&first)).unwrap();
        registry.register(Arc::clone(&second)).unwrap();

        // This event is ordered after both Registered events. Waiting for its
        // fallback result proves the capacity task consumed the registrations
        // while both workers were still Pending, making the regression
        // deterministic rather than scheduler-timing dependent.
        registry.transition_status(&sentinel_id, WorkerStatus::NotReady);
        wait_for_capacity(&mut capacity, 512).await;

        activate_registered_workers(&registry, &[first, second]).unwrap();
        wait_for_capacity(&mut capacity, 76).await;
        assert_eq!(tracker.current(), 76);
        assert_eq!(tracker.source(), CapacitySource::WorkerReported);
    }
}
