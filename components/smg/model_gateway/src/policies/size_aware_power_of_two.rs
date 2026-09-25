//! Size-aware power-of-two choices with router-local work reservations.

use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;
use rand::RngExt;
use tracing::{debug, warn};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{routers::common::header_utils::worker_url_is_allowed, worker::Worker};

pub const DEFAULT_OUTPUT_TOKEN_ESTIMATE: u64 = 4096;
const APPROX_BYTES_PER_TOKEN: u64 = 4;

/// Power-of-two choices using work reserved synchronously by this router.
///
/// Unlike engine-polled load, reservations are visible to the next routing
/// decision immediately. The estimate combines approximate input tokens with
/// a gateway-wide output estimate capped by the request's output-token limit.
#[derive(Debug)]
pub struct SizeAwarePowerOfTwoPolicy {
    output_token_estimate: u64,
    reserved_work: Mutex<HashMap<String, u64>>,
}

impl SizeAwarePowerOfTwoPolicy {
    pub fn new(output_token_estimate: u64) -> Self {
        Self {
            output_token_estimate: output_token_estimate.max(1),
            reserved_work: Mutex::new(HashMap::new()),
        }
    }

    fn estimated_work(&self, info: &SelectWorkerInfo<'_>) -> u64 {
        let input_tokens = info
            .tokens
            .map(|tokens| tokens.len() as u64)
            .or_else(|| {
                info.request_text
                    .map(|text| (text.len() as u64).div_ceil(APPROX_BYTES_PER_TOKEN).max(1))
            })
            .unwrap_or(1);
        let output_tokens = info
            .max_output_tokens
            .unwrap_or(self.output_token_estimate)
            .min(self.output_token_estimate);

        input_tokens.saturating_add(output_tokens).max(1)
    }

    /// True when router-local work reservations are materially imbalanced.
    ///
    /// One configured output estimate is the absolute noise floor, so a lone
    /// tiny request does not defeat cache affinity. Long prompts and admission
    /// bursts exceed it immediately, before backend load polling catches up.
    pub(crate) fn is_reserved_work_imbalanced(
        &self,
        workers: &[Arc<dyn Worker>],
        candidate_indices: &[usize],
        rel_threshold: f32,
    ) -> bool {
        if candidate_indices.len() < 2 {
            return false;
        }

        let reserved = self.reserved_work.lock();
        let mut min_work = u64::MAX;
        let mut max_work = 0u64;
        for &idx in candidate_indices {
            let work = reserved
                .get(workers[idx].url())
                .copied()
                .unwrap_or_default();
            min_work = min_work.min(work);
            max_work = max_work.max(work);
        }
        let min_work = if min_work == u64::MAX { 0 } else { min_work };
        max_work.saturating_sub(min_work) > self.output_token_estimate
            && (max_work as f64) > (min_work as f64 * f64::from(rel_threshold))
    }

    pub(crate) fn eligible_candidates(
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        candidate_indices: &[usize],
    ) -> Vec<usize> {
        candidate_indices
            .iter()
            .copied()
            .filter(|&idx| {
                let state = workers[idx].routing_state();
                state.healthy
                    && state.can_execute
                    && worker_url_is_allowed(info.headers, workers[idx].url())
            })
            .collect()
    }

    /// Select by size-aware P2C within an already-filtered candidate set.
    ///
    /// Cache-aware routing uses this for cold-prefix and pressure fallbacks so
    /// its outer engine-pressure filter cannot be bypassed by a second global
    /// worker scan.
    pub(crate) fn select_worker_from_candidates(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        candidate_indices: &[usize],
    ) -> Option<usize> {
        let eligible = Self::eligible_candidates(workers, info, candidate_indices);
        if eligible.is_empty() {
            return None;
        }

        let estimated_work = self.estimated_work(info);
        let (worker_idx1, worker_idx2) = if eligible.len() == 1 {
            (eligible[0], eligible[0])
        } else {
            let mut rng = rand::rng();
            let idx1 = rng.random_range(0..eligible.len());
            let idx2 = (idx1 + 1 + rng.random_range(0..eligible.len() - 1)) % eligible.len();
            (eligible[idx1], eligible[idx2])
        };

        if !info.reserve_work {
            let selected_idx = if workers[worker_idx1].load() <= workers[worker_idx2].load() {
                worker_idx1
            } else {
                worker_idx2
            };
            workers[selected_idx].increment_processed();
            return Some(selected_idx);
        }

        // Selection and reservation share one critical section so concurrent
        // admissions cannot all observe the same pre-burst load.
        let mut reserved = self.reserved_work.lock();
        let load1 = reserved
            .get(workers[worker_idx1].url())
            .copied()
            .unwrap_or_default();
        let load2 = reserved
            .get(workers[worker_idx2].url())
            .copied()
            .unwrap_or_default();
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };
        let selected_url = workers[selected_idx].url();
        let entry = reserved.entry(selected_url.to_string()).or_default();
        *entry = entry.saturating_add(estimated_work);
        let selected_load = *entry;
        drop(reserved);

        workers[selected_idx].increment_processed();
        debug!(
            worker_url = selected_url,
            estimated_work,
            reserved_work = selected_load,
            "Size-aware power-of-two selection"
        );
        Some(selected_idx)
    }

    /// Select the least-loaded member of a cached-owner set and reserve its
    /// estimated work atomically. This keeps hot prefixes balanced across all
    /// healthy owners without discarding their cache affinity.
    pub(crate) fn select_least_loaded_from_candidates(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        candidate_indices: &[usize],
    ) -> Option<usize> {
        let eligible = Self::eligible_candidates(workers, info, candidate_indices);
        if eligible.is_empty() {
            return None;
        }

        if !info.reserve_work {
            let selected_idx = eligible.into_iter().min_by_key(|&idx| {
                let state = workers[idx].routing_state();
                (state.load, state.processed, idx)
            })?;
            workers[selected_idx].increment_processed();
            return Some(selected_idx);
        }

        let estimated_work = self.estimated_work(info);
        let mut reserved = self.reserved_work.lock();
        let selected_idx = eligible.into_iter().min_by_key(|&idx| {
            let state = workers[idx].routing_state();
            (
                reserved
                    .get(workers[idx].url())
                    .copied()
                    .unwrap_or_default(),
                state.load,
                state.processed,
                idx,
            )
        })?;
        let selected_url = workers[selected_idx].url();
        let entry = reserved.entry(selected_url.to_string()).or_default();
        *entry = entry.saturating_add(estimated_work);
        let selected_load = *entry;
        drop(reserved);

        workers[selected_idx].increment_processed();
        debug!(
            worker_url = selected_url,
            estimated_work,
            reserved_work = selected_load,
            owner_count = candidate_indices.len(),
            "Size-aware cached-owner selection"
        );
        Some(selected_idx)
    }

    #[cfg(test)]
    fn reserved_for(&self, worker_url: &str) -> u64 {
        self.reserved_work
            .lock()
            .get(worker_url)
            .copied()
            .unwrap_or_default()
    }
}

impl Default for SizeAwarePowerOfTwoPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_OUTPUT_TOKEN_ESTIMATE)
    }
}

impl LoadBalancingPolicy for SizeAwarePowerOfTwoPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        self.select_worker_from_candidates(workers, info, &healthy_indices)
    }

    fn reservation_cost(&self, info: &SelectWorkerInfo<'_>) -> Option<u64> {
        info.reserve_work.then(|| self.estimated_work(info))
    }

    fn release_reservation(&self, worker_url: &str, cost: u64) {
        let mut reserved = self.reserved_work.lock();
        let Some(current) = reserved.get_mut(worker_url) else {
            warn!(worker_url, cost, "Missing size-aware work reservation");
            return;
        };
        if *current < cost {
            warn!(
                worker_url,
                cost,
                reserved_work = *current,
                "Size-aware work reservation underflow"
            );
        }
        *current = current.saturating_sub(cost);
        if *current == 0 {
            reserved.remove(worker_url);
        }
    }

    fn remove_worker(&self, url: &str) {
        self.reserved_work.lock().remove(url);
    }

    fn name(&self) -> &'static str {
        "size_aware_power_of_two"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ]
    }

    #[test]
    fn caps_output_estimate_at_model_value() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let info = SelectWorkerInfo {
            request_text: Some(&"x".repeat(4_000)),
            max_output_tokens: Some(32_000),
            reserve_work: true,
            ..Default::default()
        };

        assert_eq!(policy.estimated_work(&info), 5_000);
    }

    #[test]
    fn honors_smaller_request_output_limit() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let info = SelectWorkerInfo {
            tokens: Some(&[1; 1_000]),
            max_output_tokens: Some(500),
            reserve_work: true,
            ..Default::default()
        };

        assert_eq!(policy.estimated_work(&info), 1_500);
    }

    #[test]
    fn immediate_reservation_sends_next_request_to_other_worker() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let workers = workers();
        let info = SelectWorkerInfo {
            tokens: Some(&[1; 1_000]),
            max_output_tokens: Some(500),
            reserve_work: true,
            ..Default::default()
        };

        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();

        assert_ne!(first, second);
        assert_eq!(policy.reserved_for(workers[first].url()), 1_500);
        assert_eq!(policy.reserved_for(workers[second].url()), 1_500);
    }

    #[test]
    fn trusted_target_and_exclusions_partition_workers() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let workers = workers();
        let mut target_headers = http::HeaderMap::new();
        target_headers.insert("x-smg-target-worker-url", "http://w2:8000".parse().unwrap());
        let target_info = SelectWorkerInfo {
            headers: Some(&target_headers),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &target_info), Some(1));

        let mut excluded_headers = http::HeaderMap::new();
        excluded_headers.insert(
            "x-smg-excluded-worker-urls",
            "http://w2:8000".parse().unwrap(),
        );
        let shared_info = SelectWorkerInfo {
            headers: Some(&excluded_headers),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &shared_info), Some(0));
    }

    #[test]
    fn missing_target_fails_closed() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let workers = workers();
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "x-smg-target-worker-url",
            "http://missing:8000".parse().unwrap(),
        );
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(policy.select_worker(&workers, &info), None);
    }

    #[test]
    fn completion_releases_exact_reserved_work() {
        let policy = SizeAwarePowerOfTwoPolicy::new(4_000);
        let workers = workers();
        let info = SelectWorkerInfo {
            tokens: Some(&[1; 1_000]),
            max_output_tokens: Some(500),
            reserve_work: true,
            ..Default::default()
        };

        let selected = policy.select_worker(&workers, &info).unwrap();
        policy.release_reservation(workers[selected].url(), 1_500);

        assert_eq!(policy.reserved_for(workers[selected].url()), 0);
    }
}
