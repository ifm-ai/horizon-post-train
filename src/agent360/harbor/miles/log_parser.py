"""
Async SLURM log tailer for the training dashboard.

Tails the SLURM log file and extracts pipeline phases, metrics, errors,
and rollout progress. Updates a shared DashboardState object that the
dashboard API endpoints read from.

Usage:
    state = DashboardState()
    tailer = LogTailer("/path/to/slurm.log", state)
    asyncio.create_task(tailer.run())
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server import DashboardState

logger = logging.getLogger(__name__)

# Polling interval in seconds
POLL_INTERVAL = 2.0


# -- Compiled regex patterns --

_RE_DISPATCH = re.compile(r"Calling agent server:.*instance_id=(\S+),")
_RE_FINISHED = re.compile(
    r"Instance (\S+) finished: exit_status=(\S+), reward=([\d.]+)"
)
_RE_ROLLOUT_PROGRESS = re.compile(
    r"Rollout generation:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)"
)
_RE_WEIGHT_UPDATE_START = re.compile(r"Timer update_weights start")
_RE_WEIGHT_UPDATE_END = re.compile(r"Timer update_weights end.*elapsed: ([\d.]+)s")
_RE_WEIGHT_SYNC = re.compile(r"update_weights_from_distributed.*200 OK")
_RE_DIAG = re.compile(
    r"DIAG sample\[(\d+)\]\[(\d+)\]:.*tokens=(\d+).*resp_len=(\d+)"
)
_RE_SGLANG_THROUGHPUT = re.compile(
    r"gen throughput \(token/s\): ([\d.]+)"
)
_RE_SGLANG_RUNNING = re.compile(
    r"Decode batch.*#running-req: (\d+)"
)
_RE_SGLANG_DECODE_QUEUE = re.compile(
    r"Decode batch.*#running-req: (\d+).*#queue-req: (\d+)"
)
_RE_SGLANG_PREFILL_BATCH = re.compile(
    r"Prefill batch.*#running-req: (\d+).*#queue-req: (\d+)"
)
_RE_SGLANG_PREFILL_INFLIGHT = re.compile(
    r"Prefill batch.*#prealloc-req: (\d+).*#inflight-req: (\d+)"
)
_RE_GRAD_NORM = re.compile(r"actor/grad_norm['\"]?\s*[:=]\s*([\d.]+)")
_RE_TRAIN_STEP = re.compile(r"global_steps?:\s*(\d+)")
_RE_SESSION_NOT_FOUND = re.compile(r"session not found")
_RE_ERROR_LINE = re.compile(
    r"(?:^|\]\s*|\)\s+)(ERROR|Traceback \(most recent call last\)|FAILED)"
)
# Matches JSON-format error records emitted by harbor_server.py via the
# JsonFormatter in log_format.py. The plain-text regex above expects ERROR
# preceded by `^`, `]`, or `)` and would miss `"level":"ERROR"` (preceded
# by `"`). Both patterns exist because the SLURM combined log contains a
# mix of JSON (harbor_server) and plain text (miles, sglang, ray).
_RE_JSON_ERROR_LINE = re.compile(r'"level"\s*:\s*"(ERROR|CRITICAL)"')
_RE_INSTANCE_IN_LOG = re.compile(r"harbor\.trial\.trial\.(\S+?)(?:__\w+)?\.")
_RE_WANDB_RUN_URL = re.compile(r"View run at (https://wandb\.ai/\S+)")
_RE_WANDB_PROJECT_URL = re.compile(r"View project at (https://wandb\.ai/\S+)")

def resolve_log_path() -> str | None:
    """Determine the SLURM log file path from environment variables."""
    explicit = os.environ.get("DASHBOARD_LOG_PATH")
    if explicit and Path(explicit).exists():
        return explicit

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None

    repo_dir = os.environ.get("REPO_DIR", "")
    if not repo_dir:
        home = os.environ.get("HOME_DIR", os.path.expanduser("~"))
        repo_dir = os.path.join(home, "GitHub", "RL360")

    # Try known log name patterns
    slurm_dir = os.path.join(repo_dir, "slurm")
    for pattern in [f"harbor-async-v2-{job_id}.log", f"harbor-rl-{job_id}.log", f"viewer-test-{job_id}.log"]:
        candidate = os.path.join(slurm_dir, pattern)
        if Path(candidate).exists():
            return candidate

    # Glob fallback: any log file matching the job ID
    import glob
    matches = glob.glob(os.path.join(slurm_dir, f"*-{job_id}.log"))
    if matches:
        return matches[0]

    return None


class LogTailer:
    """Tails a log file and updates dashboard state with parsed events."""

    def __init__(self, log_path: str, state: DashboardState):
        self.log_path = log_path
        self.state = state
        self._position: int = 0
        self._rollout_id: int = 0
        self._last_dispatch_count: int = 0

    def _scan_existing_log(self) -> None:
        """Quick scan of existing log content for one-time events like wandb URL."""
        try:
            with open(self.log_path, "r", errors="replace") as f:
                # Read first 100KB (wandb URL appears early in the log)
                content = f.read(200_000)
            for line in content.splitlines():
                m = _RE_WANDB_RUN_URL.search(line)
                if m:
                    self.state.cluster_info["wandb_run_url"] = m.group(1)
                m = _RE_WANDB_PROJECT_URL.search(line)
                if m:
                    self.state.cluster_info["wandb_project_url"] = m.group(1)
                m = _RE_TRAIN_STEP.search(line)
                if m:
                    step = int(m.group(1))
                    if step > self.state.training_step:
                        self.state.training_step = step
        except Exception as e:
            logger.warning(f"Failed to scan existing log: {e}")

    async def run(self) -> None:
        """Main loop: poll for new log lines every POLL_INTERVAL seconds."""
        # Scan existing log for key one-time events (wandb URL, training step, etc.)
        # before seeking to end for live tailing
        try:
            self._scan_existing_log()
            self._position = Path(self.log_path).stat().st_size
        except OSError:
            self._position = 0

        logger.info(f"LogTailer started: {self.log_path} (pos={self._position})")

        while True:
            try:
                await self._read_new_lines()
            except Exception as e:
                logger.warning(f"LogTailer error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _read_new_lines(self) -> None:
        """Read new bytes from the log file and parse them."""
        try:
            size = Path(self.log_path).stat().st_size
        except OSError:
            return

        if size <= self._position:
            if size < self._position:
                # File was truncated (new job?), reset
                self._position = 0
            return

        # Read in a thread to avoid blocking the event loop
        def _read():
            with open(self.log_path, "r", errors="replace") as f:
                f.seek(self._position)
                data = f.read(size - self._position)
            return data

        data = await asyncio.to_thread(_read)
        self._position = size

        for line in data.splitlines():
            self._parse_line(line)

    def _parse_line(self, line: str) -> None:
        """Parse a single log line and update state."""
        # Task dispatched
        m = _RE_DISPATCH.search(line)
        if m:
            self._last_dispatch_count += 1
            self.state.pipeline_phase = "agent_execution"
            return

        # Task finished
        m = _RE_FINISHED.search(line)
        if m:
            instance_id, exit_status, reward_str = m.group(1), m.group(2), m.group(3)
            reward = float(reward_str)
            # Update metrics incrementally
            metrics = self.state.metrics
            total = metrics.get("_total_finished", 0) + 1
            total_reward = metrics.get("_total_reward", 0.0) + reward
            total_solved = metrics.get("_total_solved", 0) + (1 if reward > 0 else 0)
            metrics["_total_finished"] = total
            metrics["_total_reward"] = total_reward
            metrics["_total_solved"] = total_solved
            metrics["mean_reward"] = total_reward / total if total else 0
            metrics["solve_rate"] = total_solved / total if total else 0
            self.state._notify_subscribers({
                "type": "trial_finished_log",
                "instance_id": instance_id,
                "exit_status": exit_status,
                "reward": reward,
            })
            return

        # Rollout progress bar
        m = _RE_ROLLOUT_PROGRESS.search(line)
        if m:
            pct, done, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if done == 0 and self.state.rollout_progress.get("done", 0) > 0:
                # New rollout started
                self._rollout_id += 1
                self._last_dispatch_count = 0
            self.state.rollout_progress = {
                "done": done,
                "total": total,
                "pct": pct,
                "rollout_id": self._rollout_id,
            }
            self.state.pipeline_phase = "rollout_generation"
            self.state._notify_subscribers({
                "type": "rollout_progress",
                **self.state.rollout_progress,
            })
            return

        # Weight update
        if _RE_WEIGHT_UPDATE_START.search(line):
            self.state.pipeline_phase = "weight_update"
            self.state._notify_subscribers({"type": "pipeline_phase", "phase": "weight_update"})
            return

        m = _RE_WEIGHT_UPDATE_END.search(line)
        if m:
            self.state.metrics["last_weight_update_sec"] = float(m.group(1))
            self.state.pipeline_phase = "rollout_generation"
            self.state._notify_subscribers({"type": "pipeline_phase", "phase": "rollout_generation"})
            return

        # Training step
        m = _RE_TRAIN_STEP.search(line)
        if m:
            step = int(m.group(1))
            if step > self.state.training_step:
                self.state.training_step = step
                self.state.pipeline_phase = "training_step"
                self.state._notify_subscribers({
                    "type": "training_step",
                    "step": step,
                })
            return

        # Grad norm
        m = _RE_GRAD_NORM.search(line)
        if m:
            self.state.metrics["grad_norm"] = float(m.group(1))
            return

        # SGLang throughput (sample, not every line)
        m = _RE_SGLANG_THROUGHPUT.search(line)
        if m:
            self.state.sglang_throughput = float(m.group(1))
            return

        # SGLang PD disaggregation stats
        m = _RE_SGLANG_PREFILL_BATCH.search(line)
        if m:
            pd = self.state.metrics.setdefault("pd_stats", {})
            pd["prefill_running_reqs"] = int(m.group(1))
            pd["prefill_queue_reqs"] = int(m.group(2))
            m2 = _RE_SGLANG_PREFILL_INFLIGHT.search(line)
            if m2:
                pd["prefill_prealloc_reqs"] = int(m2.group(1))
                pd["prefill_inflight_reqs"] = int(m2.group(2))
            self.state._notify_subscribers({"type": "pd_stats", **pd})
            return

        m = _RE_SGLANG_DECODE_QUEUE.search(line)
        if m:
            pd = self.state.metrics.setdefault("pd_stats", {})
            pd["decode_running_reqs"] = int(m.group(1))
            pd["decode_queue_reqs"] = int(m.group(2))
            self.state._notify_subscribers({"type": "pd_stats", **pd})
            return

        # DIAG sample
        m = _RE_DIAG.search(line)
        if m:
            self.state.metrics.setdefault("diag_samples", [])
            self.state.metrics["diag_samples"].append({
                "group": int(m.group(1)),
                "sample": int(m.group(2)),
                "tokens": int(m.group(3)),
                "resp_len": int(m.group(4)),
            })
            # Keep only last 32
            if len(self.state.metrics["diag_samples"]) > 32:
                self.state.metrics["diag_samples"] = self.state.metrics["diag_samples"][-32:]
            return

        # WandB run URL
        m = _RE_WANDB_RUN_URL.search(line)
        if m:
            self.state.cluster_info["wandb_run_url"] = m.group(1)
            self.state._notify_subscribers({"type": "wandb_url", "url": m.group(1)})
            return

        m = _RE_WANDB_PROJECT_URL.search(line)
        if m:
            self.state.cluster_info["wandb_project_url"] = m.group(1)
            return

        # Errors
        if _RE_SESSION_NOT_FOUND.search(line):
            self.state.error_count += 1
            msg = line.strip()[-2000:]
            self.state.last_error = msg
            entry = {"time": time.time(), "msg": msg}
            m_inst = _RE_INSTANCE_IN_LOG.search(line)
            if m_inst:
                entry["instance_id"] = m_inst.group(1)
            self.state.errors.append(entry)
            self.state._notify_subscribers({"type": "error", **entry})
            return

        if _RE_ERROR_LINE.search(line) or _RE_JSON_ERROR_LINE.search(line):
            # Skip noisy SGLang/Ray internal errors
            if any(skip in line for skip in ("SGLangEngine", "repeated", "NCCL_DEBUG", "stream", "canceled by remote", "error_injection", "accelerator visible", "Sending keys:", "min_timeout_sec:", "Trajectory dumped")):
                return
            self.state.error_count += 1
            msg = line.strip()[-2000:]
            self.state.last_error = msg
            entry = {"time": time.time(), "msg": msg}
            m_inst = _RE_INSTANCE_IN_LOG.search(line)
            if m_inst:
                entry["instance_id"] = m_inst.group(1)
            self.state.errors.append(entry)
