//! Per-worker KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per gRPC worker that subscribes
//! to KV cache events and feeds them into a shared `PositionalIndexer` (one per model).
//! This enables event-driven cache-aware routing as an alternative to the approximate
//! radix tree approach.
//!
//! Lifecycle:
//! - `on_worker_added` — spawns streaming task, creates indexer if needed
//! - `on_worker_removed` — signals graceful shutdown, task cleans up indexer
//! - `stop` — signals shutdown to all tasks, clears state

use std::{
    collections::HashMap,
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::{DashMap, DashSet};
use futures::FutureExt as _;
use kv_index::{
    compute_content_hash, ApplyError, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap,
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvEventBatch,
};
use tokio::{
    sync::{oneshot, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    observability::metrics::Metrics,
    worker::{ConnectionMode, Worker, UNKNOWN_MODEL_ID},
};

/// Default jump size for new `PositionalIndexer` instances.
const DEFAULT_JUMP_SIZE: usize = 64;

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Maximum reconnection delay (caps exponential backoff).
const MAX_RECONNECT_DELAY_MS: u64 = 30_000;

/// Manages per-worker KV cache event subscriptions.
///
/// Each gRPC worker gets a dedicated tokio task that subscribes to the backend's
/// KV cache event stream and feeds events into a shared `PositionalIndexer`
/// (one per `model_id`). Workers serving the same model share the same indexer.
pub struct KvEventMonitor {
    /// Per-model positional indexers: model_id → shared indexer.
    pub(crate) indexers: DashMap<String, Arc<PositionalIndexer>>,
    /// Per-model block sizes learned from KV events or set via WorkerSpec.
    /// Used by CacheAwarePolicy to chunk request tokens at query time.
    /// Arc-wrapped so subscription tasks can update it from events.
    block_sizes: Arc<DashMap<String, usize>>,
    /// Workers whose current stream has applied at least one contiguous batch.
    event_ready_workers: Arc<DashSet<String>>,
    /// Per-worker seqlock generation. Odd means the index is being mutated;
    /// certified readers accept a claim only across the same even generation.
    event_generations: Arc<DashMap<String, u64>>,
    /// Last successfully applied batch, bounding trust on heartbeatless streams.
    event_last_batch: Arc<DashMap<String, Instant>>,
    /// Per-worker subscription handles: worker_url → subscription info.
    /// Mutex matches LoadMonitor pattern for atomic abort + remove.
    worker_handles: Mutex<HashMap<String, WorkerSubscription>>,
    /// Jump size for new PositionalIndexer instances.
    jump_size: usize,
}

/// Tracks a single worker's subscription state.
struct WorkerSubscription {
    handle: JoinHandle<()>,
    model_id: String,
    /// Signals the subscription task to shut down gracefully.
    /// The task owns its `WorkerBlockMap` and cleans up the indexer on exit.
    shutdown_tx: oneshot::Sender<()>,
}

/// Result of processing a stream connection to completion.
enum StreamResult {
    /// Stream closed normally (server-side).
    Ended,
    /// Stream produced an error.
    Error(String),
}

#[derive(Debug, PartialEq, Eq)]
enum BatchResult {
    Applied,
    EpochReset { previous: u64, received: u64 },
    GapRecovered { expected: u64, received: u64 },
}

impl KvEventMonitor {
    /// Create a new `KvEventMonitor`.
    ///
    /// `jump_size` controls the `PositionalIndexer` jump search stride.
    /// Pass `None` for the default (64).
    pub fn new(jump_size: Option<usize>) -> Self {
        let jump_size = jump_size.unwrap_or(DEFAULT_JUMP_SIZE).max(1);
        Self {
            indexers: DashMap::new(),
            block_sizes: Arc::new(DashMap::new()),
            event_ready_workers: Arc::new(DashSet::new()),
            event_generations: Arc::new(DashMap::new()),
            event_last_batch: Arc::new(DashMap::new()),
            worker_handles: Mutex::new(HashMap::new()),
            jump_size,
        }
    }

    /// Start a KV event subscription for a worker.
    ///
    /// Spawns a background tokio task that subscribes to KV cache events via
    /// server-streaming gRPC and applies them to the model's `PositionalIndexer`.
    /// Duplicate calls for the same worker URL are no-ops.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url().to_string();
        // Normalize model_id to match routing's normalize_model_key — empty → "unknown".
        let model_id = Self::normalize_model_id(worker.model_id());

        if *worker.connection_mode() == ConnectionMode::Http {
            debug!(worker_url = %url, "HTTP worker, skipping KV event subscription");
            return;
        }

        let mut handles = self.worker_handles.lock().await;
        if handles.contains_key(&url) {
            debug!(worker_url = %url, "KV event subscription already active, skipping");
            return;
        }

        let indexer = self
            .indexers
            .entry(model_id.clone())
            .or_insert_with(|| Arc::new(PositionalIndexer::new(self.jump_size)))
            .clone();

        // Seed block_size provisionally from WorkerSpec. The event stream will
        // overwrite this with the backend's actual page size once received.
        if let Some(bs) = worker.metadata().spec.kv_block_size {
            if bs > 0 {
                self.block_sizes.entry(model_id.clone()).or_insert(bs);
            } else {
                warn!(worker_url = %url, "Worker reports kv_block_size=0, ignoring");
            }
        }

        let worker = Arc::clone(worker);
        let worker_url = url.clone();
        let block_sizes = Arc::clone(&self.block_sizes);
        let event_ready_workers = Arc::clone(&self.event_ready_workers);
        let event_generations = Arc::clone(&self.event_generations);
        let event_last_batch = Arc::clone(&self.event_last_batch);
        Self::begin_generation_mutation(&event_generations, &url);
        event_ready_workers.remove(&url);
        event_last_batch.remove(&url);

        info!(
            worker_url = %url,
            model_id = %model_id,
            "Starting KV event subscription"
        );

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let loop_model_id = model_id.clone();
        let task_url = url.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "KV event monitor: runs for the lifetime of the worker, \
                      handle is stored and graceful shutdown is sent on removal"
        )]
        let handle = tokio::spawn(async move {
            let worker_id = match indexer.intern_worker(&task_url) {
                Ok(id) => id,
                Err(e) => {
                    error!(
                        worker_url = %task_url,
                        error = %e,
                        "Failed to intern worker; KV events disabled for this worker"
                    );
                    Metrics::record_kv_event_subscription_failure(&task_url, "intern_failed");
                    return;
                }
            };
            // Keep reverse membership outside the caught future so panic
            // cleanup can remove every exact membership from the shared index.
            let mut worker_blocks = WorkerBlockMap::default();
            // Catch panics here so they surface when they happen — a bare
            // JoinError would only be observed at worker removal, leaving the
            // index silently frozen for this worker until then.
            let result = std::panic::AssertUnwindSafe(Self::subscription_loop(
                worker,
                worker_url,
                Arc::clone(&indexer),
                worker_id,
                &mut worker_blocks,
                block_sizes,
                Arc::clone(&event_ready_workers),
                Arc::clone(&event_generations),
                Arc::clone(&event_last_batch),
                loop_model_id,
                shutdown_rx,
            ))
            .catch_unwind()
            .await;
            Self::begin_generation_mutation(&event_generations, &task_url);
            event_ready_workers.remove(&task_url);
            event_last_batch.remove(&task_url);
            indexer.remove_worker(worker_id, worker_blocks);
            if let Err(payload) = result {
                let msg = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .map(String::from)
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "(non-string panic)".into());
                error!(
                    worker_url = %task_url,
                    panic.message = %msg,
                    "KV event subscription task panicked; KV events from this \
                     worker no longer feed cache-aware routing"
                );
                Metrics::record_kv_event_subscription_failure(&task_url, "panic");
            }
        });

        handles.insert(
            url,
            WorkerSubscription {
                handle,
                model_id,
                shutdown_tx,
            },
        );
    }

    /// Stop the KV event subscription for a worker.
    ///
    /// Sends a graceful shutdown signal — the task cleans up its own
    /// `WorkerBlockMap` in the indexer before exiting.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        // Serialize same-URL removal and re-add. The old task must finish
        // clearing its worker ID before a replacement begins publishing.
        let mut handles = self.worker_handles.lock().await;
        let Some(sub) = handles.remove(worker_url) else {
            return;
        };

        info!(worker_url = %worker_url, "Stopping KV event subscription");
        // Signal graceful shutdown — task cleans up its worker_blocks in the indexer.
        let _ = sub.shutdown_tx.send(());
        // Panics are caught inside the task; a JoinError here (abort or a
        // panic that escaped the guard) must still be surfaced, not discarded.
        if let Err(e) = sub.handle.await {
            error!(
                worker_url = %worker_url,
                error = %e,
                "KV event subscription task failed"
            );
            Metrics::record_kv_event_subscription_failure(worker_url, "join_error");
        }

        // Re-check under lock whether this was the last worker for the model.
        // Must re-acquire lock after shutdown to avoid TOCTOU with concurrent
        // on_worker_added that may have added a new worker for the same model
        // between our first lock release and this point.
        let should_remove_indexer = !handles.values().any(|other| other.model_id == sub.model_id);

        if should_remove_indexer {
            self.indexers.remove(&sub.model_id);
            self.block_sizes.remove(&sub.model_id);
        }
        Self::begin_generation_mutation(&self.event_generations, worker_url);
        self.event_ready_workers.remove(worker_url);
        self.event_last_batch.remove(worker_url);
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        let subscriptions: HashMap<String, WorkerSubscription> = {
            let mut handles = self.worker_handles.lock().await;
            std::mem::take(&mut *handles)
        };

        if !subscriptions.is_empty() {
            info!(
                count = subscriptions.len(),
                "Stopping all KV event subscriptions"
            );
            for (url, sub) in subscriptions {
                debug!(worker_url = %url, "Stopping KV event subscription");
                let _ = sub.shutdown_tx.send(());
                if let Err(e) = sub.handle.await {
                    error!(
                        worker_url = %url,
                        error = %e,
                        "KV event subscription task failed"
                    );
                    Metrics::record_kv_event_subscription_failure(&url, "join_error");
                }
            }
        }

        self.indexers.clear();
        self.block_sizes.clear();
        self.event_ready_workers.clear();
        self.event_generations.clear();
        self.event_last_batch.clear();
    }

    /// Get the indexer for a model (used by `CacheAwarePolicy` for queries).
    pub fn get_indexer(&self, model_id: &str) -> Option<Arc<PositionalIndexer>> {
        self.indexers.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the block size for a model (learned from events or set via `set_block_size`).
    pub fn block_size(&self, model_id: &str) -> Option<usize> {
        self.block_sizes.get(model_id).map(|v| *v)
    }

    /// Set the block size for a model (e.g. from WorkerSpec during registration).
    /// Does not overwrite a value already learned from events.
    pub fn set_block_size(&self, model_id: &str, block_size: usize) {
        self.block_sizes
            .entry(model_id.to_string())
            .or_insert(block_size);
    }

    /// Capture the even stream/index generation for a certified query.
    pub fn certified_generation(&self, worker_url: &str, max_age: Duration) -> Option<u64> {
        let generation = self.event_generations.get(worker_url).map(|entry| *entry)?;
        let observed_at = self.event_last_batch.get(worker_url).map(|entry| *entry)?;
        (generation.is_multiple_of(2)
            && self.event_ready_workers.contains(worker_url)
            && Instant::now().saturating_duration_since(observed_at) <= max_age)
            .then_some(generation)
    }

    /// Verify a claim at the selection linearization point.
    pub fn certified_generation_unchanged(
        &self,
        worker_url: &str,
        generation: u64,
        max_age: Duration,
    ) -> bool {
        generation.is_multiple_of(2)
            && self.event_ready_workers.contains(worker_url)
            && self
                .event_last_batch
                .get(worker_url)
                .is_some_and(|observed| {
                    Instant::now().saturating_duration_since(*observed) <= max_age
                })
            && self
                .event_generations
                .get(worker_url)
                .is_some_and(|current| *current == generation)
    }

    #[cfg(test)]
    pub(crate) fn mark_event_stream_ready_for_test(&self, worker_url: &str) {
        Self::begin_generation_mutation(&self.event_generations, worker_url);
        self.event_last_batch
            .insert(worker_url.to_string(), Instant::now());
        self.event_ready_workers.insert(worker_url.to_string());
        Self::finish_generation_mutation(&self.event_generations, worker_url);
    }

    #[cfg(test)]
    pub(crate) fn mark_event_stream_not_ready_for_test(&self, worker_url: &str) {
        Self::begin_generation_mutation(&self.event_generations, worker_url);
        self.event_ready_workers.remove(worker_url);
        self.event_last_batch.remove(worker_url);
    }

    #[cfg(test)]
    pub(crate) fn age_event_stream_for_test(&self, worker_url: &str, age: Duration) {
        self.event_last_batch.insert(
            worker_url.to_string(),
            Instant::now().checked_sub(age).unwrap_or_else(Instant::now),
        );
    }

    /// Check if any subscription is running.
    pub async fn is_running(&self) -> bool {
        !self.worker_handles.lock().await.is_empty()
    }

    /// Normalize model_id to match routing's `normalize_model_key`.
    /// Empty model IDs map to UNKNOWN_MODEL_ID for consistent keying.
    fn normalize_model_id(model_id: &str) -> String {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            model_id.to_string()
        }
    }

    fn begin_generation_mutation(generations: &DashMap<String, u64>, worker_url: &str) {
        generations
            .entry(worker_url.to_string())
            .and_modify(|generation| {
                if generation.is_multiple_of(2) {
                    *generation = generation.wrapping_add(1);
                }
            })
            .or_insert(1);
    }

    fn finish_generation_mutation(generations: &DashMap<String, u64>, worker_url: &str) {
        generations
            .entry(worker_url.to_string())
            .and_modify(|generation| {
                if !generation.is_multiple_of(2) {
                    *generation = generation.wrapping_add(1);
                }
            })
            .or_insert(2);
    }

    // -----------------------------------------------------------------------
    // Subscription loop
    // -----------------------------------------------------------------------

    /// Learn `block_size` from the first `KvBlock` in a stored event.
    ///
    /// Called once per model when the first stored event arrives, providing
    /// ground truth from the backend. `CacheAwarePolicy` uses this to chunk
    /// request tokens into blocks for overlap scoring.
    ///
    /// Overwrites any provisional value seeded from `WorkerSpec` since the
    /// event stream reflects the backend's actual page size.
    fn learn_block_size(
        block_sizes: &DashMap<String, usize>,
        model_id: &str,
        learned: &mut bool,
        batch: &KvEventBatch,
    ) {
        if *learned {
            return;
        }
        for event in &batch.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if let Some(block) = stored.blocks.first() {
                    if block.block_size > 0 {
                        let bs = block.block_size as usize;
                        block_sizes.insert(model_id.to_string(), bs);
                        info!(
                            model_id = %model_id,
                            block_size = bs,
                            "Learned block_size from KV event"
                        );
                        *learned = true;
                        return;
                    }
                }
            }
        }
    }

    /// Main subscription loop for a single worker.
    ///
    /// Owns the `WorkerBlockMap` for this worker and cleans it up on exit.
    /// Exits when `shutdown_rx` fires or the backend returns `Unimplemented`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the live stream task owns explicit per-worker certification state"
    )]
    async fn subscription_loop(
        worker: Arc<dyn Worker>,
        worker_url: String,
        indexer: Arc<PositionalIndexer>,
        worker_id: u32,
        worker_blocks: &mut WorkerBlockMap,
        block_sizes: Arc<DashMap<String, usize>>,
        event_ready_workers: Arc<DashSet<String>>,
        event_generations: Arc<DashMap<String, u64>>,
        event_last_batch: Arc<DashMap<String, Instant>>,
        model_id: String,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        let mut last_seq: u64 = 0;
        let mut seen_batch = false;
        let mut reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
        let mut block_size_learned = false;

        /// Sleep with shutdown check. Returns `true` if shutdown was signaled.
        macro_rules! sleep_or_shutdown {
            ($delay:expr, $rx:expr) => {
                tokio::select! {
                    _ = tokio::time::sleep($delay) => false,
                    _ = &mut *$rx => true,
                }
            };
        }

        macro_rules! invalidate_and_clear {
            () => {{
                Self::begin_generation_mutation(&event_generations, &worker_url);
                event_ready_workers.remove(&worker_url);
                event_last_batch.remove(&worker_url);
                indexer.apply_cleared(worker_id, worker_blocks);
                return;
            }};
        }

        loop {
            let grpc_client = match worker.get_grpc_client().await {
                Ok(Some(client)) => client,
                Ok(None) => {
                    // HTTP workers are filtered in on_worker_added, so this should
                    // be unreachable. Retry defensively rather than exiting and
                    // leaving a stale entry in worker_handles.
                    warn!(
                        worker_url = %worker_url,
                        delay_ms = reconnect_delay_ms,
                        "Worker has no gRPC client yet, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        invalidate_and_clear!();
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to get gRPC client, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        invalidate_and_clear!();
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let stream = match grpc_client.subscribe_kv_events(last_seq).await {
                Ok(stream) => {
                    info!(
                        worker_url = %worker_url,
                        start_seq = last_seq,
                        "KV event stream connected"
                    );
                    reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                    stream
                }
                Err(e) => {
                    // If the backend doesn't implement SubscribeKvEvents (e.g. vLLM),
                    // stop retrying — this RPC will never succeed.
                    if e.code() == tonic::Code::Unimplemented {
                        warn!(
                            worker_url = %worker_url,
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this worker"
                        );
                        invalidate_and_clear!();
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to subscribe to KV events, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        invalidate_and_clear!();
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let on_batch = |batch: &KvEventBatch| {
                Self::learn_block_size(&block_sizes, &model_id, &mut block_size_learned, batch);
            };
            let stream_result = tokio::select! {
                result = Self::process_stream(
                    stream, &worker_url, worker_id, &indexer,
                    worker_blocks, &mut last_seq, &mut seen_batch,
                    &event_ready_workers, &event_generations, &event_last_batch,
                    on_batch,
                ) => result,
                _ = &mut shutdown_rx => {
                    invalidate_and_clear!();
                }
            };
            // Production KV subscriptions are live-only. A reconnect cannot
            // prove that the engine process or publisher epoch stayed the
            // same, so discard every old positive before subscribing again.
            Self::begin_generation_mutation(&event_generations, &worker_url);
            event_ready_workers.remove(&worker_url);
            event_last_batch.remove(&worker_url);
            Self::reset_stream_epoch(
                &indexer,
                worker_id,
                worker_blocks,
                &mut last_seq,
                &mut seen_batch,
            );

            match stream_result {
                StreamResult::Ended => {
                    info!(
                        worker_url = %worker_url,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream ended, reconnecting"
                    );
                    // Backoff to avoid tight reconnect loop if server keeps
                    // closing the stream cleanly (e.g., rolling connections).
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        invalidate_and_clear!();
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
                StreamResult::Error(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream error, reconnecting"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        invalidate_and_clear!();
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stream processing + proto conversion
    // -----------------------------------------------------------------------

    /// Process batches from a single stream connection.
    #[expect(
        clippy::too_many_arguments,
        reason = "the stream processor updates one explicit worker epoch atomically"
    )]
    async fn process_stream(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
        last_seq: &mut u64,
        seen_batch: &mut bool,
        event_ready_workers: &DashSet<String>,
        event_generations: &DashMap<String, u64>,
        event_last_batch: &DashMap<String, Instant>,
        mut on_batch: impl FnMut(&KvEventBatch),
    ) -> StreamResult {
        use tokio_stream::StreamExt;

        while let Some(result) = stream.next().await {
            let batch = match result {
                Ok(batch) => batch,
                Err(e) => return StreamResult::Error(e.to_string()),
            };

            Self::begin_generation_mutation(event_generations, worker_url);
            Self::process_batch(
                &batch,
                worker_url,
                worker_id,
                indexer,
                worker_blocks,
                last_seq,
                seen_batch,
                &mut on_batch,
            );
            event_last_batch.insert(worker_url.to_string(), Instant::now());
            event_ready_workers.insert(worker_url.to_string());
            Self::finish_generation_mutation(event_generations, worker_url);
        }

        StreamResult::Ended
    }

    /// Apply one KV event batch, recovering live-only streams after a sequence gap.
    #[expect(
        clippy::too_many_arguments,
        reason = "batch recovery mutates one explicit worker epoch and its test callback"
    )]
    fn process_batch(
        batch: &KvEventBatch,
        worker_url: &str,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
        last_seq: &mut u64,
        seen_batch: &mut bool,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchResult {
        // The production stream is live-only. A non-increasing sequence on an
        // otherwise-connected bridge is an engine publisher epoch reset, not
        // replay. Clear first so equal/lower restart collisions cannot retain
        // stale positives.
        if *seen_batch && batch.sequence_number <= *last_seq {
            let previous = *last_seq;
            warn!(
                worker_url = %worker_url,
                last_seq = *last_seq,
                received = batch.sequence_number,
                "KV publisher sequence reset; cleared stale cache state"
            );
            indexer.apply_cleared(worker_id, worker_blocks);
            for event in &batch.events {
                Self::apply_event(event, worker_id, indexer, worker_blocks);
            }
            on_batch(batch);
            *last_seq = batch.sequence_number;
            *seen_batch = true;
            return BatchResult::EpochReset {
                previous,
                received: batch.sequence_number,
            };
        }

        // Some engine bridges expose a live-only event stream even though the
        // gRPC request carries a replay cursor. Reconnecting such a stream after
        // one dropped event starts at the current publisher position, so retrying
        // the old cursor can never fill the gap. Clear this worker's stale view
        // and resume from the first live batch instead. The approximate token tree
        // remains available while new event-driven ownership is learned.
        let expected = last_seq.saturating_add(1);
        let gap = (*seen_batch && batch.sequence_number > expected)
            .then_some((expected, batch.sequence_number));
        if let Some((expected, received)) = gap {
            warn!(
                worker_url = %worker_url,
                expected = expected,
                received = received,
                "Sequence gap detected; cleared stale cache state and resumed live stream"
            );
            Metrics::record_kv_event_sequence_gap_recovery(worker_url);
            indexer.apply_cleared(worker_id, worker_blocks);
        }

        for event in &batch.events {
            Self::apply_event(event, worker_id, indexer, worker_blocks);
        }
        on_batch(batch);
        *last_seq = batch.sequence_number;
        *seen_batch = true;

        match gap {
            Some((expected, received)) => BatchResult::GapRecovered { expected, received },
            None => BatchResult::Applied,
        }
    }

    fn reset_stream_epoch(
        indexer: &PositionalIndexer,
        worker_id: u32,
        worker_blocks: &mut WorkerBlockMap,
        last_seq: &mut u64,
        seen_batch: &mut bool,
    ) {
        indexer.apply_cleared(worker_id, worker_blocks);
        *last_seq = 0;
        *seen_batch = false;
    }

    /// Apply a single KV cache event to the indexer.
    fn apply_event(
        event: &KvCacheEvent,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let Some(ref data) = event.data else {
            return;
        };

        match data {
            kv_cache_event::Data::Stored(stored) => {
                Self::apply_stored(stored, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Removed(removed) => {
                Self::apply_removed(removed, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Cleared(_) => {
                indexer.apply_cleared(worker_id, worker_blocks);
            }
        }
    }

    /// Convert proto `KvBlocksStored` and apply to the indexer.
    fn apply_stored(
        stored: &KvBlocksStored,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let blocks: Vec<StoredBlock> = stored.blocks.iter().map(convert_kv_block).collect();

        let parent_seq_hash = stored.parent_block_hash.map(SequenceHash::from);

        match indexer.apply_stored(worker_id, &blocks, parent_seq_hash, worker_blocks) {
            Ok(()) => {}
            Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) => {
                // Re-rooting an orphan from a live-only cold start would put a
                // child at position zero and manufacture a false prefix hit.
                warn!(
                    worker_id = worker_id,
                    parent_block_hash = ?stored.parent_block_hash,
                    "Skipped KV store event with unknown parent"
                );
            }
        }
    }

    /// Convert proto `KvBlocksRemoved` and apply to the indexer.
    fn apply_removed(
        removed: &KvBlocksRemoved,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let seq_hashes: Vec<SequenceHash> = removed
            .block_hashes
            .iter()
            .map(|&h| SequenceHash::from(h))
            .collect();

        indexer.apply_removed(worker_id, &seq_hashes, worker_blocks);
    }
}

/// Convert a proto `KvBlock` to a kv-index `StoredBlock`.
fn convert_kv_block(block: &KvBlock) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: compute_content_hash(&block.token_ids),
    }
}

impl Drop for KvEventMonitor {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.worker_handles.try_lock() {
            for (_, sub) in handles.drain() {
                let _ = sub.shutdown_tx.send(());
                sub.handle.abort(); // Can't await in Drop, abort as fallback
            }
        }
    }
}

impl fmt::Debug for KvEventMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvEventMonitor")
            .field("models", &self.indexers.len())
            .field("block_sizes", &self.block_sizes.len())
            .field("jump_size", &self.jump_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Proto → kv-index conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_kv_block() {
        let block = KvBlock {
            block_hash: 42,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_convert_kv_block_negative_hash() {
        let block = KvBlock {
            block_hash: -1,
            token_ids: vec![10, 20],
            block_size: 2,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
    }

    #[test]
    fn test_convert_kv_block_empty_tokens() {
        let block = KvBlock {
            block_hash: 100,
            token_ids: vec![],
            block_size: 0,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash::from(100i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[]));
    }

    // -----------------------------------------------------------------------
    // apply_event integration with PositionalIndexer
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_stored_no_parent() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
            ],
            parent_block_hash: None,
        };

        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_apply_stored_with_parent() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored1 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored1, w1, &indexer, &mut wb);

        let stored2 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 2,
                token_ids: vec![50, 60, 70, 80],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: Some(1),
        };
        KvEventMonitor::apply_stored(&stored2, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_apply_stored_unknown_parent_never_becomes_false_root() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://new-worker:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        // A live-only stream may start with a child whose parent predates this
        // gateway. It must not be re-rooted at position zero.
        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: Some(999),
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
        assert!(wb.is_empty());
    }

    #[test]
    fn test_apply_removed() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
            ],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);

        let removed = KvBlocksRemoved {
            block_hashes: vec![2],
            cache_level: None,
        };
        KvEventMonitor::apply_removed(&removed, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_cleared_event() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        indexer.apply_cleared(w1, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_sequence_gap_clears_stale_worker_state_before_resuming() {
        let indexer = PositionalIndexer::new(64);
        let worker_url = "http://w1:8000";
        let worker_id = indexer.intern_worker(worker_url).unwrap();
        let mut worker_blocks = WorkerBlockMap::default();
        let mut last_seq = 1;
        let mut seen_batch = true;

        let stale = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stale, worker_id, &indexer, &mut worker_blocks);
        assert_eq!(indexer.current_size(), 1);

        let live = KvEventBatch {
            sequence_number: 3,
            timestamp: 0.0,
            events: vec![KvCacheEvent {
                event_id: 2,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 2,
                        token_ids: vec![50, 60, 70, 80],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: Some(999),
                })),
            }],
            dp_rank: None,
        };

        let result = KvEventMonitor::process_batch(
            &live,
            worker_url,
            worker_id,
            &indexer,
            &mut worker_blocks,
            &mut last_seq,
            &mut seen_batch,
            &mut |_| {},
        );

        assert_eq!(
            result,
            BatchResult::GapRecovered {
                expected: 2,
                received: 3
            }
        );
        assert_eq!(last_seq, 3);
        assert_eq!(indexer.current_size(), 0);
        assert!(!worker_blocks.contains_key(&SequenceHash::from(1i64)));
        assert!(!worker_blocks.contains_key(&SequenceHash::from(2i64)));

        let contiguous = KvEventBatch {
            sequence_number: 4,
            timestamp: 0.0,
            events: vec![KvCacheEvent {
                event_id: 3,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 3,
                        token_ids: vec![90, 100, 110, 120],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: None,
                })),
            }],
            dp_rank: None,
        };
        let result = KvEventMonitor::process_batch(
            &contiguous,
            worker_url,
            worker_id,
            &indexer,
            &mut worker_blocks,
            &mut last_seq,
            &mut seen_batch,
            &mut |_| {},
        );

        assert_eq!(result, BatchResult::Applied);
        assert_eq!(last_seq, 4);
        assert_eq!(indexer.current_size(), 1);
        assert!(worker_blocks.contains_key(&SequenceHash::from(3i64)));
    }

    #[test]
    fn sequence_zero_then_gap_clears_old_positive() {
        let indexer = PositionalIndexer::new(64);
        let worker_url = "http://w1:8000";
        let worker_id = indexer.intern_worker(worker_url).unwrap();
        let mut worker_blocks = WorkerBlockMap::default();
        let mut last_seq = 0;
        let mut seen_batch = false;
        let root = KvEventBatch {
            sequence_number: 0,
            timestamp: 0.0,
            events: vec![KvCacheEvent {
                event_id: 1,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 1,
                        token_ids: vec![10, 20, 30, 40],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: None,
                })),
            }],
            dp_rank: None,
        };
        assert_eq!(
            KvEventMonitor::process_batch(
                &root,
                worker_url,
                worker_id,
                &indexer,
                &mut worker_blocks,
                &mut last_seq,
                &mut seen_batch,
                &mut |_| {},
            ),
            BatchResult::Applied
        );
        assert!(seen_batch);
        assert_eq!(last_seq, 0);
        assert_eq!(indexer.current_size(), 1);

        // Missing sequence 1 may have removed that root. Sequence 2 must
        // clear it even though the last observed value was the valid zero.
        let jumped = KvEventBatch {
            sequence_number: 2,
            timestamp: 0.0,
            events: Vec::new(),
            dp_rank: None,
        };
        assert_eq!(
            KvEventMonitor::process_batch(
                &jumped,
                worker_url,
                worker_id,
                &indexer,
                &mut worker_blocks,
                &mut last_seq,
                &mut seen_batch,
                &mut |_| {},
            ),
            BatchResult::GapRecovered {
                expected: 1,
                received: 2,
            }
        );
        assert_eq!(indexer.current_size(), 0);
        assert!(worker_blocks.is_empty());
    }

    #[test]
    fn equal_and_lower_sequence_resets_clear_old_positive() {
        fn root_batch(sequence_number: u64, block_hash: i64) -> KvEventBatch {
            KvEventBatch {
                sequence_number,
                timestamp: 0.0,
                events: vec![KvCacheEvent {
                    event_id: sequence_number,
                    data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                        blocks: vec![KvBlock {
                            block_hash,
                            token_ids: vec![1, 2, 3, 4],
                            block_size: 4,
                            lora_id: None,
                            cache_level: None,
                        }],
                        parent_block_hash: None,
                    })),
                }],
                dp_rank: None,
            }
        }

        for (old_sequence, reset_sequence) in [(0, 0), (5, 1)] {
            let indexer = PositionalIndexer::new(64);
            let worker_url = "http://w1:8000";
            let worker_id = indexer.intern_worker(worker_url).unwrap();
            let mut worker_blocks = WorkerBlockMap::default();
            let mut last_seq = 0;
            let mut seen_batch = false;
            KvEventMonitor::process_batch(
                &root_batch(old_sequence, 1),
                worker_url,
                worker_id,
                &indexer,
                &mut worker_blocks,
                &mut last_seq,
                &mut seen_batch,
                &mut |_| {},
            );
            assert_eq!(indexer.current_size(), 1);

            let reset = KvEventBatch {
                sequence_number: reset_sequence,
                timestamp: 0.0,
                events: Vec::new(),
                dp_rank: None,
            };
            assert_eq!(
                KvEventMonitor::process_batch(
                    &reset,
                    worker_url,
                    worker_id,
                    &indexer,
                    &mut worker_blocks,
                    &mut last_seq,
                    &mut seen_batch,
                    &mut |_| {},
                ),
                BatchResult::EpochReset {
                    previous: old_sequence,
                    received: reset_sequence,
                }
            );
            assert_eq!(indexer.current_size(), 0);
            assert!(worker_blocks.is_empty());
        }
    }

    #[test]
    fn reconnect_reset_clears_index_and_invalidates_generation() {
        let monitor = KvEventMonitor::new(Some(4));
        let indexer = PositionalIndexer::new(64);
        let worker_url = "http://w1:8000";
        let worker_id = indexer.intern_worker(worker_url).unwrap();
        let mut worker_blocks = WorkerBlockMap::default();
        indexer
            .apply_stored(
                worker_id,
                &[StoredBlock {
                    seq_hash: SequenceHash(1),
                    content_hash: compute_content_hash(&[1, 2, 3, 4]),
                }],
                None,
                &mut worker_blocks,
            )
            .unwrap();
        monitor.mark_event_stream_ready_for_test(worker_url);
        let generation = monitor
            .certified_generation(worker_url, Duration::from_secs(30))
            .unwrap();
        let mut last_seq = 9;
        let mut seen_batch = true;

        monitor.mark_event_stream_not_ready_for_test(worker_url);
        KvEventMonitor::reset_stream_epoch(
            &indexer,
            worker_id,
            &mut worker_blocks,
            &mut last_seq,
            &mut seen_batch,
        );

        assert_eq!(indexer.current_size(), 0);
        assert!(worker_blocks.is_empty());
        assert_eq!(last_seq, 0);
        assert!(!seen_batch);
        assert!(!monitor.certified_generation_unchanged(
            worker_url,
            generation,
            Duration::from_secs(30)
        ));
    }

    #[test]
    fn test_apply_event_dispatch_stored() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 42,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                }],
                parent_block_hash: None,
            })),
        };

        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_event_dispatch_removed() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored_event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 1,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                }],
                parent_block_hash: None,
            })),
        };
        KvEventMonitor::apply_event(&stored_event, w1, &indexer, &mut wb);

        let removed_event = KvCacheEvent {
            event_id: 2,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: vec![1],
                cache_level: None,
            })),
        };
        KvEventMonitor::apply_event(&removed_event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_apply_event_dispatch_cleared() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 1,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 1,
                        token_ids: vec![1, 2, 3, 4],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: None,
                })),
            },
            w1,
            &indexer,
            &mut wb,
        );

        // Clear
        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 2,
                data: Some(kv_cache_event::Data::Cleared(
                    smg_grpc_client::common_proto::KvCacheCleared {},
                )),
            },
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_apply_event_no_data() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: None,
        };
        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_monitor_new() {
        let monitor = KvEventMonitor::new(None);
        assert!(!monitor.is_running().await);
    }

    #[tokio::test]
    async fn test_monitor_new_clamps_zero_jump_size() {
        let monitor = KvEventMonitor::new(Some(0));
        assert_eq!(monitor.jump_size, 1);
    }

    #[tokio::test]
    async fn test_get_indexer_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        assert!(monitor.get_indexer("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_stop_empty_monitor() {
        let monitor = KvEventMonitor::new(None);
        monitor.stop().await;
    }

    #[tokio::test]
    async fn test_on_worker_removed_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        monitor.on_worker_removed("http://nonexistent:8000").await;
    }

    // -----------------------------------------------------------------------
    // block_size learning
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_block_size() {
        let monitor = KvEventMonitor::new(None);

        // Initially no block_size
        assert!(monitor.block_size("llama").is_none());

        // Set it
        monitor.set_block_size("llama", 32);
        assert_eq!(monitor.block_size("llama"), Some(32));

        // set_block_size doesn't overwrite existing value
        monitor.set_block_size("llama", 64);
        assert_eq!(monitor.block_size("llama"), Some(32));
    }

    #[tokio::test]
    async fn test_stop_clears_block_sizes() {
        let monitor = KvEventMonitor::new(None);
        monitor.set_block_size("llama", 16);
        assert_eq!(monitor.block_size("llama"), Some(16));

        monitor.stop().await;
        assert!(monitor.block_size("llama").is_none());
    }
}
