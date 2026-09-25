# Modified for release: configurable deployment paths and public source references.
"""
FastAPI server wrapping Harbor for generalized agent-environment orchestration.

Provides a single ``/run`` endpoint that handles any task type (SWE-bench,
Terminal-Bench, custom datasets, etc.) through Harbor's unified Trial API.
Harbor handles Docker orchestration, agent execution, and grading — the
server is task-type agnostic.

Requires:
    - Harbor installed: pip install harbor-framework
    - Prepared task dirs under HARBOR_TASKS_DIR (via adapters or prepare_harbor_tasks.py)

Usage:
    python server.py --port 11000 --max-concurrent 8
"""

import argparse
import asyncio
import collections
import dataclasses
import fcntl
import json
import logging
import multiprocessing as mp
import os
import re
import time
import shutil
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent360.harbor.miles.agent_config import (
    MINI_SWE_AGENT_NAME,
    AgentConfigError,
    deep_merge_dicts,
    validate_mini_swe_agent_config,
)
from agent360.harbor.miles import log_parser
from agent360.harbor.miles.log_format import install_json_formatter

logger = logging.getLogger(__name__)

# Module-level inflight counter for the /run semaphore. Maintained explicitly
# so we don't depend on asyncio.Semaphore._value (private API).
_run_inflight: int = 0


# -- Persistence config (dashboard-persistence) --
_SNAPSHOT_DIR: Path | None = None
_TRIALS_JSONL: Path | None = None
_STATE_JSON: Path | None = None
_S3_BUCKET: str = os.getenv("RL360_S3_BUCKET", "")
_VIEWER_URL: str = os.getenv("VIEWER_URL", "")
_INGEST_TOKEN: str = os.getenv("INGEST_TOKEN", "")
_JOB_USER: str = os.getenv(
    "JOB_USER", os.getenv("SLURM_JOB_USER", os.getenv("USER", "unknown"))
)
_JOB_UID: int | None = None
try:
    _JOB_UID = int(os.getenv("JOB_UID", os.getenv("SLURM_JOB_UID", "")))
except ValueError:
    pass

# -- Dashboard state --


@dataclass
class TrialRecord:
    """Tracks the lifecycle of a single Harbor trial."""

    instance_id: str
    sample_idx: int = 0
    phase: str = "queued"  # queued | env_setup | agent_running | verifying | done | error | cancelled
    reward: float = 0.0
    exit_status: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    phase_times: dict = field(default_factory=dict)
    agent_metrics: dict = field(default_factory=dict)
    trial_dir: str | None = None
    error_msg: str = ""
    rollout_id: int = 0
    turn_count: int = 0
    env_setup_sec: float = 0.0
    persistent_dir: str | None = None

    def _update_rollout_progress(self):
        total = self._rollout_dispatched
        done = self._rollout_finished
        self.rollout_progress["total"] = total
        self.rollout_progress["done"] = done
        self.rollout_progress["pct"] = int(100 * done / max(1, total))
        self._notify_subscribers({"type": "rollout_progress", **self.rollout_progress})

    def to_dict(self) -> dict:
        now = time.time()
        d = dataclasses.asdict(self)
        d["duration_sec"] = (
            round((self.end_time or now) - self.start_time, 1) if self.start_time else 0
        )
        if self.phase in ("queued", "env_setup", "agent_running", "verifying"):
            phase_start = (
                self.phase_times.get(self.phase, (now,))[0]
                if self.phase_times.get(self.phase)
                else self.start_time
            )
            d["phase_duration_sec"] = round(now - phase_start, 1)
        else:
            d["phase_duration_sec"] = 0
        d["has_trajectory"] = bool(
            self.trial_dir and Path(self.trial_dir, "agent", "trajectory.json").exists()
        )
        return d


class DashboardState:
    """Shared mutable state for the dashboard. All access is on the asyncio event loop."""

    def __init__(self):
        self.trials: collections.deque[TrialRecord] = collections.deque(maxlen=500)
        self.active_trials: dict[str, TrialRecord] = {}
        self.pipeline_phase: str = "initializing"
        self.rollout_progress: dict = {"done": 0, "total": 0, "pct": 0, "rollout_id": 0}
        self.training_step: int = 0
        self.metrics: dict = {}
        self.error_count: int = 0
        self.last_error: str = ""
        self.errors: collections.deque = collections.deque(maxlen=50)
        self.sglang_throughput: float = 0.0
        self.subscribers: list[asyncio.Queue] = []
        self._rollout_dispatched: int = 0
        self._rollout_finished: int = 0
        self.cluster_info: dict = {}

    def _notify_subscribers(self, event: dict) -> None:
        for q in self.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Throttle: log at most once every 60s with the cumulative
                # drop count. Avoids spamming when one slow SSE client falls
                # behind for a sustained period.
                self._queue_full_drops = getattr(self, "_queue_full_drops", 0) + 1
                last = getattr(self, "_queue_full_last_log", 0.0)
                now = time.time()
                if now - last >= 60.0:
                    self._queue_full_last_log = now
                    logger.warning(
                        f"dashboard_queue_full dropped={self._queue_full_drops} subs={len(self.subscribers)}",
                        extra={
                            "event": "dashboard_queue_full",
                            "dropped_count": self._queue_full_drops,
                            "subscribers_count": len(self.subscribers),
                        },
                    )

    def _update_rollout_progress(self):
        total = self._rollout_dispatched
        done = self._rollout_finished
        self.rollout_progress["total"] = total
        self.rollout_progress["done"] = done
        self.rollout_progress["pct"] = int(100 * done / max(1, total))
        self._notify_subscribers({"type": "rollout_progress", **self.rollout_progress})

    def to_dict(self) -> dict:
        return {
            "pipeline_phase": self.pipeline_phase,
            "rollout_progress": self.rollout_progress,
            "training_step": self.training_step,
            "metrics": {k: v for k, v in self.metrics.items() if not k.startswith("_")},
            "error_count": self.error_count,
            "last_error": self.last_error,
            "errors": [{"time": e["time"], "msg": e["msg"]} for e in self.errors],
            "sglang_throughput": self.sglang_throughput,
            "cluster_info": self.cluster_info,
            "active_trials": [r.to_dict() for r in self.active_trials.values()],
            "recent_trials": [r.to_dict() for r in reversed(self.trials)],
        }


_dashboard_state = DashboardState()

_semaphore: asyncio.Semaphore | None = None

# Active /run tasks in this worker process. /abort_all cancels these tasks,
# which propagates asyncio.CancelledError into Harbor Trial.run(). Harbor then
# emits TrialEvent.CANCEL and finalizes the environment via BaseEnvironment.stop().
_active_run_tasks: dict[int, dict[str, Any]] = {}
_active_run_task_counter: int = 0

# Set only in worker child processes launched by main().
_WORKER_ID: int = int(os.getenv("HARBOR_WORKER_ID", "0"))


def _parse_session_base_url(api_base: str | None) -> str | None:
    if not api_base:
        return None

    base = api_base.rstrip("/")
    marker = "/sessions/"
    idx = base.rfind(marker)
    if idx == -1:
        return None

    session_id = base[idx + len(marker) :].split("/")[0]
    if not session_id:
        return None

    return base[: idx + len(marker)] + session_id


async def _close_session(session_base_url: str) -> None:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.delete(session_base_url)

        logger.info(
            f"session_closed url={session_base_url}",
            extra={"event": "session_closed", "session_base_url": session_base_url},
        )
    except Exception as e:
        logger.warning(
            f"session_close_failed url={session_base_url}: {e}",
            extra={
                "event": "session_close_failed",
                "session_base_url": session_base_url,
                "error": str(e),
            },
        )


async def _abort_local_active_trials(reason: str = "abort_all") -> dict[str, Any]:
    active_items = list(_active_run_tasks.items())
    cancelled: list[dict[str, Any]] = []
    already_done: list[dict[str, Any]] = []
    close_coros = []

    for run_task_id, info in active_items:
        task = info.get("task")
        instance_id = str(info.get("instance_id", ""))
        session_base_url = info.get("session_base_url")

        if not isinstance(task, asyncio.Task):
            continue

        item = {"run_task_id": run_task_id, "instance_id": instance_id}

        if task.done():
            already_done.append(item)
            continue

        task.cancel()

        if isinstance(session_base_url, str) and session_base_url:
            item["session_close_requested"] = True
            item["session_base_url"] = session_base_url
            close_coros.append(_close_session(session_base_url))
        else:
            item["session_close_requested"] = False

        cancelled.append(item)

    if close_coros:
        await asyncio.gather(*close_coros, return_exceptions=True)

    logger.warning(
        f"{reason}: worker_id={_WORKER_ID} worker_pid={os.getpid()} "
        f"aborted_trials={len(cancelled)} already_done={len(already_done)} "
        f"session_close_requested={len(close_coros)}",
        extra={
            "event": reason,
            "worker_id": _WORKER_ID,
            "worker_pid": os.getpid(),
            "aborted_trials": len(cancelled),
            "already_done": len(already_done),
            "session_close_requested": len(close_coros),
            "aborted_instances": [item["instance_id"] for item in cancelled],
            "already_done_instances": [item["instance_id"] for item in already_done],
        },
    )

    _dashboard_state._notify_subscribers(
        {
            "type": reason,
            "worker_id": _WORKER_ID,
            "worker_pid": os.getpid(),
            "aborted_trials": len(cancelled),
            "already_done": len(already_done),
            "session_close_requested": len(close_coros),
        }
    )

    return {
        "worker_id": _WORKER_ID,
        "worker_pid": os.getpid(),
        "aborted_trials": len(cancelled),
        "already_done": len(already_done),
        "session_close_requested": len(close_coros),
        "aborted": cancelled,
        "already_done_items": already_done,
    }


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _semaphore
    # Ensure JSON formatter is installed on every worker. main() also calls
    # this for the parent process, but uvicorn workers re-import the app
    # without re-entering main(), so we re-install here. install_json_formatter
    # is idempotent.
    install_json_formatter()

    max_concurrent = int(
        os.getenv(
            "AGENT_MAX_CONCURRENT", os.getenv("SWE_AGENT_MAX_CONCURRENT", "40000")
        )
    )
    _semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(
        "semaphore_initialized",
        extra={"event": "semaphore_initialized", "max_concurrent": max_concurrent},
    )

    # Start log tailer for dashboard
    log_path = log_parser.resolve_log_path()
    if log_path:
        asyncio.create_task(log_parser.LogTailer(log_path, _dashboard_state).run())
        logger.info(
            f"Dashboard log tailer started: {log_path}",
            extra={"event": "log_tailer_started"},
        )
    else:
        logger.warning(
            "No SLURM log found for dashboard (set DASHBOARD_LOG_PATH or SLURM_JOB_ID)",
            extra={"event": "log_tailer_missing"},
        )

    # Start periodic trajectory saver for crash resilience
    asyncio.create_task(_periodic_trajectory_saver())
    asyncio.create_task(_periodic_rss_logger())

    # Populate cluster info from environment
    _dashboard_state.cluster_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_nodelist": os.getenv("SLURM_JOB_NODELIST", ""),
        "hostname": os.getenv("HOSTNAME", ""),
        "harbor_tasks_dir": os.getenv("HARBOR_TASKS_DIR", ""),
        "wandb_project": os.getenv("WANDB_PROJECT", ""),
        "wandb_group": os.getenv("WANDB_GROUP", ""),
        "wandb_run_url": "",
        "wandb_project_url": "",
    }

    # -- Snapshot directory for crash resilience (dashboard-persistence) --
    global _SNAPSHOT_DIR, _TRIALS_JSONL, _STATE_JSON
    shared_dir = os.getenv("SHARED_DIR", "./outputs")
    job_id = os.getenv("SLURM_JOB_ID", "local")
    _SNAPSHOT_DIR = Path(shared_dir) / "snapshots" / job_id
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    # Workers share one trials file; appends are flock-serialized. The
    # periodic state dump is single-worker only: each worker only sees its
    # own trials, and the exit trap already captures state.json via the
    # dashboard API.
    multi_worker = int(os.getenv("HARBOR_NUM_WORKERS", "1")) > 1
    _TRIALS_JSONL = _SNAPSHOT_DIR / "trials.jsonl"
    _STATE_JSON = None if multi_worker else _SNAPSHOT_DIR / "state.json"

    _config = {
        "job_id": job_id,
        "cluster": os.getenv("SLURM_CLUSTER_NAME", "local"),
        "model": os.getenv("HF_CHECKPOINT", ""),
        "submitted_by": _JOB_USER,
        "submitted_uid": _JOB_UID,
        "wandb_project": os.getenv("WANDB_PROJECT", ""),
        "wandb_group": os.getenv("WANDB_GROUP", ""),
        "slurm_nodelist": os.getenv("SLURM_JOB_NODELIST", ""),
        "num_nodes": int(os.getenv("SLURM_JOB_NUM_NODES", "0")),
        "started_at": time.time(),
    }
    (_SNAPSHOT_DIR / "config.json").write_text(json.dumps(_config, indent=2))
    _register_job_with_viewer(_config)
    asyncio.create_task(_periodic_state_dumper())
    asyncio.create_task(_periodic_heartbeat())

    yield


app = FastAPI(title="Agent Environment Server (Harbor)", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_OPENAI_API_KEY_PLACEHOLDER = "dummy"


class RunRequest(BaseModel):
    base_url: str
    model: str
    sampling_params: dict[str, Any]
    api_key: str = _OPENAI_API_KEY_PLACEHOLDER

    instance_id: str = ""
    sample_idx: int | None = None
    agent_name: str
    max_seq_len: int | None = None
    llm_timeout_sec: float

    # Extra kwargs forwarded to the agent's LLM client (e.g. LiteLLM extra_body).
    # Used for W-TITO on the agentic /v1/chat/completions path so RL rollouts
    # can request completion token IDs and routed-expert indices back from the
    # SGLang worker. See terminus_2.py:__init__ for consumption.
    llm_call_kwargs: dict[str, Any] = {}
    agent_config: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = {}
    eval_report: dict[str, Any] = {}


def _api_key_or_placeholder(api_key: str | None) -> str:
    # The Miles session tracer is an OpenAI-compatible local endpoint. It does
    # not authenticate these calls, but LiteLLM/OpenAI clients still expect a
    # non-empty API key.
    return api_key or _OPENAI_API_KEY_PLACEHOLDER


_TIMEOUT_EXCEPTION_MAP = {
    "AgentTimeoutError": "AgentTimeout",
    "VerifierTimeoutError": "VerifierTimeout",
    "EnvironmentStartTimeoutError": "EnvStartTimeout",
}

_TERMINUS_HOST_AGENTS = {"terminus-2", "terminus-1", "terminus"}
_HOST_PROCESS_AGENTS = _TERMINUS_HOST_AGENTS | {
    "mini-swe-agent-external",
}

_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_VERIFIER_LOG_BYTES = 200_000
_VERIFIER_OUTPUT_FILES = [
    ("reward_txt", "reward.txt", "Reward"),
    ("reward_json", "reward.json", "Reward JSON"),
    ("test_stdout", "test-stdout.txt", "Test stdout"),
    ("test_stderr", "test-stderr.txt", "Test stderr"),
    ("ctrf", "ctrf.json", "CTRF report"),
]


def _extract_exit_status(result) -> str:
    """Derive exit status from Harbor TrialResult.

    Priority:
      1. Trial-level exception (timeout / env failure) → specific name.
      2. Verifier ran → "Submitted" (agent submitted a solution).
      3. Agent's own exit status from trajectory metadata.
      4. "Unknown" fallback.
    """
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "") or "AgentError"
        return _TIMEOUT_EXCEPTION_MAP.get(exc_type, exc_type)
    if getattr(result, "verifier_result", None) is not None:
        return "Submitted"
    agent_result = getattr(result, "agent_result", None)
    if agent_result is not None:
        metadata = getattr(agent_result, "metadata", None) or {}
        status = metadata.get("exit_status", "")
        if status:
            return status
    return "Unknown"


def _timing_duration_sec(timing) -> float | None:
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if started and finished:
        return (finished - started).total_seconds()
    return None


def _extract_reward(result) -> tuple[float, dict[str, Any]]:
    """Extract scalar reward and full eval report from Harbor TrialResult.

    Looks for the ``"reward"`` key first, then falls back to the first value
    in the rewards dict. Works with both ``reward.txt`` and ``reward.json``.
    """
    vr = getattr(result, "verifier_result", None)
    if vr is None:
        return 0.0, {}
    rewards = getattr(vr, "rewards", None) or {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))
    return reward, dict(rewards)


def _extract_metrics(result) -> dict[str, Any]:
    """Extract agent metrics from Harbor TrialResult."""
    metrics: dict[str, Any] = {}
    try:
        ar = getattr(result, "agent_result", None)
        if ar is not None:
            for field in ("n_input_tokens", "n_output_tokens", "cost_usd"):
                val = getattr(ar, field, None)
                if val is not None:
                    metrics[field] = val
            agent_meta = getattr(ar, "metadata", None)
            if isinstance(agent_meta, dict):
                metrics.update(agent_meta)

        agent_timing = getattr(result, "agent_execution", None)
        if agent_timing is not None:
            dur = _timing_duration_sec(agent_timing)
            if dur is not None:
                metrics["agent_run_time"] = dur

        verifier_timing = getattr(result, "verifier", None)
        if verifier_timing is not None:
            dur = _timing_duration_sec(verifier_timing)
            if dur is not None:
                metrics["eval_time"] = dur
    except Exception as e:
        logger.warning(f"Failed to extract metrics: {e}", exc_info=True)
    return metrics


def _error_response(exit_status: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "exit_status": exit_status,
        "agent_metrics": {},
        "eval_report": {},
    }


def _build_agent_kwargs_and_env(
    request: RunRequest,
) -> tuple[dict[str, Any], dict[str, str]]:
    api_key = _api_key_or_placeholder(request.api_key)

    is_host_process_agent = request.agent_name in _HOST_PROCESS_AGENTS
    is_mini_swe_external = request.agent_name == MINI_SWE_AGENT_NAME
    request_agent_config = {} if request.agent_config is None else request.agent_config
    if request_agent_config != {} and not is_mini_swe_external:
        raise AgentConfigError(
            "agent_config is only supported for agent_name='mini-swe-agent-external'."
        )

    agent_kwargs: dict[str, Any] = (
        validate_mini_swe_agent_config(request_agent_config)
        if is_mini_swe_external
        else {}
    )

    if not is_mini_swe_external and (
        "hosted_vllm" in request.model or "openai" in request.model
    ):
        agent_kwargs["model_info"] = {
            "max_input_tokens": int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768")),
            "max_output_tokens": int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "8192")),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }

    # Honor per-call max_tokens from miles' rollout sampling_params (set
    # via --rollout-max-response-len). Without this cap, terminus-2's
    # litellm calls generate up to max_output_tokens (8192), taking
    # ~15 min per turn at iter-scale throughput and 502-ing the router.
    sampling_max_tokens = (
        request.sampling_params.get("max_tokens") if request.sampling_params else None
    )
    if sampling_max_tokens:
        model_info = agent_kwargs.get("model_info")
        if isinstance(model_info, dict):
            model_info["max_output_tokens"] = min(
                model_info.get("max_output_tokens", sampling_max_tokens),
                int(sampling_max_tokens),
            )

    if request.max_seq_len:
        model_info = agent_kwargs.get("model_info")
        if isinstance(model_info, dict):
            model_info["max_output_tokens"] = min(
                model_info.get("max_output_tokens", request.max_seq_len),
                request.max_seq_len,
            )

    if request.agent_name in _TERMINUS_HOST_AGENTS:
        agent_kwargs["api_base"] = request.base_url
        agent_kwargs["api_key"] = api_key
        agent_kwargs["enable_summarize"] = False
        agent_kwargs["interleaved_thinking"] = True
        agent_kwargs["record_episode_debug"] = False
        llm_kwargs = agent_kwargs.get("llm_kwargs")
        if not isinstance(llm_kwargs, dict):
            llm_kwargs = {}
        llm_kwargs["timeout"] = request.llm_timeout_sec
        agent_kwargs["llm_kwargs"] = llm_kwargs
        # Forward llm_call_kwargs (e.g. extra_body with
        # return_completion_token_ids / return_routed_experts) into
        # Terminus-2, which passes them through to every LLM call.
        if request.llm_call_kwargs:
            agent_kwargs["llm_call_kwargs"] = dict(request.llm_call_kwargs)
    elif is_mini_swe_external:
        model_kwargs: dict[str, Any] = dict(request.llm_call_kwargs)
        extra_body = dict(model_kwargs.get("extra_body") or {})
        extra_body.update(
            {
                "return_token_ids": True,
                "return_prompt_token_ids": True,
                "return_completion_token_ids": True,
                "return_routed_experts": True,
            }
        )
        model_kwargs.update(
            {
                "api_base": request.base_url,
                "api_key": api_key,
                "logprobs": True,
                "extra_body": extra_body,
            }
        )
        model_kwargs.pop("base_url", None)
        for key in ("max_tokens", "temperature", "top_p"):
            model_kwargs[key] = request.sampling_params[key]
        model_kwargs["timeout"] = request.llm_timeout_sec
        agent_kwargs["model_overrides"] = {
            "model_kwargs": model_kwargs,
            "instance_id": request.instance_id,
        }
        # Cap agent turns. The agent stops with exit_status=LimitsExceeded once
        # its LLM-call count reaches step_limit; 0 leaves it unbounded.
        step_limit = int(os.getenv("AGENT_STEP_LIMIT", "0"))
        if step_limit > 0:
            agent_kwargs["step_limit"] = step_limit

    if is_host_process_agent:
        agent_env = {
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": request.base_url,
        }
    else:
        agent_env = {
            "OPENAI_API_BASE": request.base_url,
            "OPENAI_API_KEY": api_key,
            "HOSTED_VLLM_API_BASE": request.base_url,
            "HOSTED_VLLM_API_KEY": api_key,
            "MSWEA_COST_TRACKING": "ignore_errors",
        }

    return agent_kwargs, agent_env


def _get_persistent_dir(record: TrialRecord) -> Path | None:
    """Get the persistent shared storage path for a trial's artifacts."""
    output_dir = os.getenv("TRAJECTORY_OUTPUT_DIR")
    if not output_dir:
        return None
    wandb_project = os.getenv("WANDB_PROJECT", "")
    wandb_group = os.getenv("WANDB_GROUP", "")
    if wandb_project and wandb_group:
        base = Path(output_dir) / wandb_project / wandb_group
    else:
        job_id = os.getenv("SLURM_JOB_ID", "unknown")
        base = Path(output_dir) / job_id
    return base / f"{record.instance_id}_{record.sample_idx}"


def _copy_verifier_outputs(src: Path, dest: Path) -> None:
    verifier_dir = src / "verifier"
    if not verifier_dir.exists():
        return
    for _, filename, _ in _VERIFIER_OUTPUT_FILES:
        src_file = verifier_dir / filename
        try:
            if src_file.exists():
                shutil.copy2(src_file, dest / filename)
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(f"Failed to copy verifier output {src_file}: {e}")


def _save_trajectory_incremental(record: TrialRecord) -> None:
    """Copy trajectory (incrementally updated by Harbor) to persistent storage.

    Called both periodically for active trials and at completion.
    Safe to call multiple times; overwrites previous copy.
    """
    dest = _get_persistent_dir(record)
    if not dest or not record.trial_dir:
        return
    try:
        import shutil

        src = Path(record.trial_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Copy trajectory (incrementally updated by Harbor after each turn)
        traj_src = src / "agent" / "trajectory.json"
        if traj_src.exists():
            shutil.copy2(traj_src, dest / "trajectory.json")

        # Copy trial log
        log_src = src / "trial.log"
        if log_src.exists():
            shutil.copy2(log_src, dest / "trial.log")

        # Copy result.json if exists (only at completion)
        result_src = src / "result.json"
        if result_src.exists():
            shutil.copy2(result_src, dest / "result.json")

        # Copy detailed verifier scores
        scores_src = src / "verifier" / "detailed_scores.json"
        if scores_src.exists():
            shutil.copy2(scores_src, dest / "detailed_scores.json")
        _copy_verifier_outputs(src, dest)

        # Copy asciinema terminal recording
        cast_src = src / "agent" / "recording.cast"
        if cast_src.exists():
            shutil.copy2(cast_src, dest / "recording.cast")
    except Exception as e:
        logger.warning(f"Failed to save trajectory incrementally: {e}")


def _save_trajectory_final(record: TrialRecord) -> None:
    """Final save: copy all artifacts and write metadata. Update trial_dir to persistent path."""
    _save_trajectory_incremental(record)
    dest = _get_persistent_dir(record)
    if not dest:
        return
    try:
        meta = {
            "instance_id": record.instance_id,
            "sample_idx": record.sample_idx,
            "reward": record.reward,
            "exit_status": record.exit_status,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "rollout_id": record.rollout_id,
            "agent_metrics": record.agent_metrics,
            "wandb_run_url": _dashboard_state.cluster_info.get("wandb_run_url", ""),
        }
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
        record.persistent_dir = str(dest)
        logger.debug(f"Saved trajectory to {dest}")
    except Exception as e:
        logger.warning(f"Failed to finalize trajectory save: {e}")


def _update_inflight_metrics(record: TrialRecord) -> None:
    """Read token counts from an in-flight trial's native trajectory file."""
    if not record.trial_dir:
        return
    traj_path = Path(record.trial_dir) / "agent" / "mini-swe-agent.trajectory.json"
    if not traj_path.exists():
        return
    try:
        data = json.loads(traj_path.read_text())
        messages = data.get("messages", data.get("history", []))
        n_input = n_output = 0
        for m in messages:
            usage = ((m.get("extra") or {}).get("response") or {}).get("usage", {})
            n_input += usage.get("prompt_tokens", 0)
            n_output += usage.get("completion_tokens", 0)
        if n_input or n_output:
            record.agent_metrics["n_input_tokens"] = n_input
            record.agent_metrics["n_output_tokens"] = n_output
            record.agent_metrics["_inflight"] = True
    except Exception:
        pass


async def _periodic_trajectory_saver():
    """Background task: copy active trial trajectories to shared storage every 30s."""
    while True:
        await asyncio.sleep(30)
        for record in list(_dashboard_state.active_trials.values()):
            if record.trial_dir and record.phase in ("agent_running", "verifying"):
                _update_inflight_metrics(record)
                _save_trajectory_incremental(record)


def _self_rss_gib() -> float:
    """Current process RSS in GiB, read from /proc/self/status (VmRSS)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)  # kB -> GiB
    return 0.0


async def _periodic_rss_logger():
    """Log this worker's RSS every 30s so per-worker memory growth is traceable."""
    while True:
        await asyncio.sleep(30)
        logger.info(
            f"harbor_rss pid={os.getpid()} rss_gib={_self_rss_gib():.2f} "
            f"inflight={_run_inflight} active_trials={len(_dashboard_state.active_trials)}",
            extra={"event": "harbor_rss"},
        )


async def _periodic_state_dumper():
    """Dump full dashboard state to Weka every 60s for crash resilience."""
    while True:
        await asyncio.sleep(60)
        if _STATE_JSON:
            try:
                state = _dashboard_state.to_dict()
                _STATE_JSON.write_text(json.dumps(state, default=str))
            except Exception as e:
                logger.warning(f"Periodic state dump failed: {e}")


def _post_to_viewer(
    endpoint: str, payload: dict, op_name: str, timeout: float = 10.0
) -> bool:
    """POST JSON to the viewer ingest API. Returns True on HTTP 200.

    Logs a warning on any non-200 that includes the Location header if present,
    so auth-proxy redirects (e.g., Cloudflare Access) are self-evident instead
    of printing an empty body. No-op (returns False) when VIEWER_URL or
    INGEST_TOKEN is unset.
    """
    if not _VIEWER_URL or not _INGEST_TOKEN:
        return False
    try:
        import httpx

        resp = httpx.post(
            f"{_VIEWER_URL}{endpoint}",
            json=payload,
            headers={"Authorization": f"Bearer {_INGEST_TOKEN}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True
        location = resp.headers.get("location", "")
        detail = f"location={location}" if location else resp.text[:200]
        logger.warning(f"{op_name} failed ({resp.status_code}): {detail}")
        return False
    except Exception as e:
        logger.warning(f"{op_name} POST failed: {e}")
        return False


async def _periodic_heartbeat():
    """Send heartbeat to viewer every 60s so watchdog knows we are alive."""
    while True:
        await asyncio.sleep(60)
        payload = {"job_id": os.getenv("SLURM_JOB_ID", "local")}
        wandb_url = _dashboard_state.cluster_info.get("wandb_run_url", "")
        if wandb_url:
            payload["wandb_url"] = wandb_url
            # Extract entity, project, run ID from URL: https://wandb.ai/entity/project/runs/run_id
            m = re.match(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)", wandb_url)
            if m:
                payload["wandb_project"] = f"{m.group(1)}/{m.group(2)}"
                payload["wandb_run_id"] = m.group(3)
        _post_to_viewer("/api/ingest/heartbeat", payload, "Heartbeat")


def _persist_trial(record) -> None:
    """Append trial to JSONL, POST to viewer API, upload to S3. All best-effort."""
    trial_dict = record.to_dict()
    job_id = os.getenv("SLURM_JOB_ID", "local")
    trial_dict["job_id"] = job_id

    # 1. Append to trials.jsonl (crash-safe local backup)
    if _TRIALS_JSONL:
        try:
            with open(_TRIALS_JSONL, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(trial_dict, default=str) + "\n")
        except Exception as e:
            logger.warning(f"JSONL append failed: {e}")

    # 2. POST to viewer API (viewer writes to Postgres)
    prefix = f"trajectories/{job_id}/{record.instance_id}_{record.sample_idx}"
    trial_data = {
        "job_id": job_id,
        "instance_id": record.instance_id,
        "sample_idx": record.sample_idx,
        "rollout_id": record.rollout_id,
        "training_step": _dashboard_state.training_step,
        "phase": record.phase,
        "reward": record.reward,
        "exit_status": record.exit_status,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "duration_sec": trial_dict.get("duration_sec"),
        "turn_count": record.turn_count,
        "error_msg": record.error_msg,
        "agent_metrics": record.agent_metrics,
    }
    if _S3_BUCKET and record.trial_dir:
        trial_data["s3_trajectory"] = f"{prefix}/trajectory.json"
        trial_data["s3_trial_log"] = f"{prefix}/trial.log"
    _post_to_viewer("/api/ingest/trial", trial_data, "Viewer trial ingest")

    # 3. Upload trajectory to S3 (direct from SLURM node)
    if _S3_BUCKET and record.trial_dir:
        try:
            import boto3

            s3_client = boto3.client("s3")
            trial_path = Path(record.trial_dir)
            prefix = f"trajectories/{job_id}/{record.instance_id}_{record.sample_idx}"
            for fname, s3name in [
                ("agent/trajectory.json", "trajectory.json"),
                ("trial.log", "trial.log"),
                ("result.json", "result.json"),
                ("agent/recording.cast", "recording.cast"),
                ("verifier/detailed_scores.json", "detailed_scores.json"),
                ("verifier/reward.txt", "reward.txt"),
                ("verifier/reward.json", "reward.json"),
                ("verifier/test-stdout.txt", "test-stdout.txt"),
                ("verifier/test-stderr.txt", "test-stderr.txt"),
                ("verifier/ctrf.json", "ctrf.json"),
            ]:
                local = trial_path / fname
                try:
                    if local.exists():
                        s3_client.upload_file(
                            str(local), _S3_BUCKET, f"{prefix}/{s3name}"
                        )
                except FileNotFoundError:
                    continue
        except Exception as e:
            logger.warning(f"S3 upload failed for {record.instance_id}: {e}")


def _mark_trial_cancelled(
    *,
    record_key: str,
    record: TrialRecord | None,
    instance_id: str,
    sample_idx: int,
    reason: str = "Trial cancelled",
    final_save: bool = False,
    cleanup_local: bool = False,
) -> None:
    """Mark a trial cancelled in dashboard/persistence.

    Safe to call more than once. Harbor emits TrialEvent.CANCEL before its
    finalizer completes, so the hook should call this with final_save=False.
    The surrounding CancelledError handler calls it again with final_save=True
    after Trial.run() has unwound.
    """
    if record is None:
        return

    now = time.time()
    old_phase = record.phase
    if old_phase in record.phase_times:
        phase_start = record.phase_times.get(old_phase, (now, None))[0]
        record.phase_times[old_phase] = (phase_start, now)

    record.phase = "cancelled"
    record.exit_status = "Cancelled"
    record.error_msg = reason
    record.end_time = record.end_time or now

    was_active = bool(
        record_key and _dashboard_state.active_trials.pop(record_key, None) is not None
    )
    if was_active:
        _dashboard_state.trials.append(record)
        _dashboard_state._rollout_finished += 1
        _dashboard_state._update_rollout_progress()
        _dashboard_state._notify_subscribers(
            {
                "type": "trial_cancelled",
                "key": record_key,
                "instance_id": instance_id,
                "sample_idx": sample_idx,
            }
        )

    if final_save:
        _save_trajectory_final(record)
    else:
        _save_trajectory_incremental(record)
    _persist_trial(record)

    if cleanup_local and record.trial_dir:
        try:
            shutil.rmtree(Path(record.trial_dir))
        except Exception as cleanup_e:
            logger.warning(
                f"Failed to clean up cancelled trial dir: {cleanup_e}",
                extra={
                    "event": "cleanup_after_cancel_failed",
                    "instance_id": instance_id,
                    "sample_idx": sample_idx,
                    "trial_dir": record.trial_dir,
                },
            )


def _register_job_with_viewer(config: dict) -> None:
    """Register this job with the viewer API. Best-effort."""
    job_data = {
        "job_id": config["job_id"],
        "cluster": config["cluster"],
        "submitted_by": config["submitted_by"],
        "submitted_uid": config.get("submitted_uid"),
        "model": config.get("model", ""),
        "wandb_project": config.get("wandb_project", ""),
        "nodes": [config.get("slurm_nodelist", "")],
        "num_nodes": config.get("num_nodes", 0),
        "config": config,
        "status": "live",
    }
    if _post_to_viewer("/api/ingest/job", job_data, "Job registration"):
        logger.info(f"Registered job {config['job_id']} via viewer API")


async def _run_trial(request: RunRequest) -> dict[str, Any]:
    """Run a Harbor trial for a single task instance.

    Task-type agnostic — all differentiation (environment, grading harness)
    is encoded in the Harbor task directory's 4 files.
    """
    # Mock-Trial gate for laptop testing / unit tests. When the env var is
    # set, swap the lazy imports for a stub that simulates phase transitions
    # without needing harbor-framework or Daytona.
    # See _test_mock_trial.py for the schedule and configurable timings.
    import faulthandler
    import signal
    import sys
    from agent360.harbor.miles._test_mock_trial import is_mock_enabled

    if is_mock_enabled():
        from agent360.harbor.miles._test_mock_trial import MockTrial as Trial
        from agent360.harbor.miles._test_mock_trial import MockTrialEvent as TrialEvent

        # AgentConfig/EnvironmentConfig/TaskConfig/TrialConfig: mock returns
        # a plain dict so the rest of the code path doesn't need to construct
        # real harbor config objects.
        AgentConfig = EnvironmentConfig = TaskConfig = TrialConfig = dict
    else:
        try:
            from harbor.models.trial.config import (
                AgentConfig,
                EnvironmentConfig,
                TaskConfig,
                TrialConfig,
            )
            from harbor.trial.hooks import TrialEvent
            from harbor.trial.trial import Trial
        except ImportError:
            logger.error(
                "Harbor not installed. Install with: pip install harbor-framework",
                extra={"event": "harbor_import_failed"},
            )
            return _error_response("ImportError")

    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    record_key = ""
    record = None

    try:
        if not request.instance_id:
            logger.error("Empty instance_id", extra={"event": "invalid_input"})
            return _error_response("InvalidInstanceId")

        raw_id = request.instance_id
        if not _SAFE_INSTANCE_ID.match(raw_id):
            logger.error(
                f"Invalid instance_id rejected: {raw_id!r}",
                extra={"event": "invalid_input", "instance_id": raw_id},
            )
            return _error_response("InvalidInstanceId")

        tasks_dir = Path(os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks")).resolve()
        tasks_dir_str = str(tasks_dir)
        task_path = os.path.normpath(os.path.join(tasks_dir_str, raw_id))
        if not task_path.startswith(tasks_dir_str):
            logger.error(
                f"Path traversal blocked: {raw_id!r}",
                extra={"event": "path_traversal_blocked", "instance_id": raw_id},
            )
            return _error_response("InvalidInstanceId")

        if not os.path.exists(task_path):
            logger.error(
                f"Task directory not found: {task_path}",
                extra={
                    "event": "task_not_found",
                    "instance_id": raw_id,
                    "task_path": task_path,
                },
            )
            return _error_response("TaskNotFound")

        task_path = Path(task_path)
        agent_kwargs, agent_env = _build_agent_kwargs_and_env(request)

        model_name = request.model

        config = TrialConfig(
            timeout_multiplier=1.5,
            # Pin the agent budget at a flat 3600s regardless of each task's
            # task.toml timeout: override_timeout_sec replaces the base, and a
            # 1.0 agent multiplier keeps it unscaled (verifier/setup stay 1.5x).
            # Must stay <= miles' AGENT_TIMEOUT_SECS (swe_agent_function.py),
            # which wraps the whole /run call — if this budget outlives that
            # wrapper, the client gives up while Harbor is still running the
            # trial, and the eventual response has no one left to deliver to.
            agent_timeout_multiplier=1.0,
            task=TaskConfig(path=task_path),
            agent=AgentConfig(
                name=request.agent_name,
                model_name=model_name,
                override_timeout_sec=3600,
                env=agent_env,
                kwargs=agent_kwargs,
            ),
            environment=EnvironmentConfig(
                type=os.getenv("HARBOR_ENV_TYPE", "daytona"),
                delete=os.getenv("HARBOR_DELETE_CONTAINERS", "false").lower()
                in ("true", "1", "t"),
                kwargs={
                    k: v
                    for k, v in {
                        "s3_bucket": os.getenv("HARBOR_ENV_S3_BUCKET"),
                        "s3_region": os.getenv("HARBOR_ENV_S3_REGION"),
                        "registry_url": os.getenv("HARBOR_ENV_REGISTRY_URL"),
                        "inject_prebuilt_env_files": os.getenv(
                            "HARBOR_ENV_INJECT_PREBUILT_FILES", "false"
                        ).lower()
                        not in ("false", "0", "no"),
                    }.items()
                    if v is not None
                },
            ),
        )

        # -- Dashboard: register trial and lifecycle hooks --
        if request.sample_idx is None:
            raise ValueError("sample_idx is required")

        sample_idx = request.sample_idx

        record_key = f"{raw_id}_{sample_idx}"
        record = TrialRecord(
            instance_id=raw_id,
            sample_idx=sample_idx,
            phase="queued",
            start_time=time.time(),
            rollout_id=_dashboard_state.rollout_progress.get("rollout_id", 0),
        )
        _dashboard_state.active_trials[record_key] = record
        _dashboard_state._rollout_dispatched += 1
        _dashboard_state._update_rollout_progress()

        t_create_start = time.monotonic()
        trial = await Trial.create(config=config)
        record.trial_dir = str(Path(trial.trial_dir).resolve())
        logger.info(
            f"trial_create_done instance={raw_id} sample_idx={sample_idx}",
            extra={
                "event": "trial_create_done",
                "instance_id": raw_id,
                "sample_idx": sample_idx,
                "duration_secs": round(time.monotonic() - t_create_start, 3),
                "trial_dir": record.trial_dir,
            },
        )

        # Register Harbor lifecycle hooks for per-trajectory phase tracking
        async def _on_start(
            event, _r=record, _k=record_key, _iid=raw_id, _sidx=sample_idx
        ):
            _r.phase = "env_setup"
            _r.phase_times["env_setup"] = (time.time(), None)
            _dashboard_state._notify_subscribers(
                {"type": "phase_change", "key": _k, "phase": "env_setup"}
            )
            logger.info(
                f"phase_env_setup instance={_iid} sample_idx={_sidx}",
                extra={
                    "event": "phase_env_setup",
                    "instance_id": _iid,
                    "sample_idx": _sidx,
                    "phase": "env_setup",
                },
            )

        async def _on_agent_start(
            event, _r=record, _k=record_key, _iid=raw_id, _sidx=sample_idx
        ):
            env_start = _r.phase_times.get("env_setup", (time.time(),))[0]
            _r.phase_times["env_setup"] = (env_start, time.time())
            _r.env_setup_sec = round(time.time() - env_start, 1)
            _r.phase = "agent_running"
            _r.phase_times["agent_running"] = (time.time(), None)
            _dashboard_state._notify_subscribers(
                {
                    "type": "phase_change",
                    "key": _k,
                    "phase": "agent_running",
                    "env_setup_sec": _r.env_setup_sec,
                }
            )
            logger.info(
                f"phase_agent_running instance={_iid} sample_idx={_sidx} env_setup_sec={_r.env_setup_sec}",
                extra={
                    "event": "phase_agent_running",
                    "instance_id": _iid,
                    "sample_idx": _sidx,
                    "phase": "agent_running",
                    "env_setup_sec": _r.env_setup_sec,
                    "phase_duration_sec": _r.env_setup_sec,
                },
            )

        async def _on_verify_start(
            event, _r=record, _k=record_key, _iid=raw_id, _sidx=sample_idx
        ):
            agent_start = _r.phase_times.get("agent_running", (time.time(),))[0]
            agent_dur = round(time.time() - agent_start, 3)
            _r.phase_times["agent_running"] = (agent_start, time.time())
            _r.phase = "verifying"
            _r.phase_times["verifying"] = (time.time(), None)
            _dashboard_state._notify_subscribers(
                {"type": "phase_change", "key": _k, "phase": "verifying"}
            )
            logger.info(
                f"phase_verifying instance={_iid} sample_idx={_sidx} agent_running_dur={agent_dur}s",
                extra={
                    "event": "phase_verifying",
                    "instance_id": _iid,
                    "sample_idx": _sidx,
                    "phase": "verifying",
                    "phase_duration_sec": agent_dur,
                },
            )

        async def _on_end(
            event, _r=record, _k=record_key, _iid=raw_id, _sidx=sample_idx
        ):
            verifying_dur = 0.0
            if _r.phase == "verifying":
                vstart = _r.phase_times.get("verifying", (time.time(),))[0]
                _r.phase_times["verifying"] = (vstart, time.time())
                verifying_dur = round(time.time() - vstart, 3)
            logger.info(
                f"phase_end instance={_iid} sample_idx={_sidx} verifying_dur={verifying_dur}s",
                extra={
                    "event": "phase_end",
                    "instance_id": _iid,
                    "sample_idx": _sidx,
                    "phase": "end",
                    "phase_duration_sec": verifying_dur,
                },
            )

        async def _on_cancel(
            event, _r=record, _k=record_key, _iid=raw_id, _sidx=sample_idx
        ):
            _mark_trial_cancelled(
                record_key=_k,
                record=_r,
                instance_id=_iid,
                sample_idx=_sidx,
                reason="Trial cancelled",
                final_save=False,
                cleanup_local=False,
            )
            logger.info(
                f"phase_cancelled instance={_iid} sample_idx={_sidx}",
                extra={
                    "event": "phase_cancelled",
                    "instance_id": _iid,
                    "sample_idx": _sidx,
                    "phase": "cancelled",
                },
            )

        trial.add_hook(TrialEvent.START, _on_start)
        trial.add_hook(TrialEvent.AGENT_START, _on_agent_start)
        trial.add_hook(TrialEvent.VERIFICATION_START, _on_verify_start)
        trial.add_hook(TrialEvent.CANCEL, _on_cancel)
        trial.add_hook(TrialEvent.END, _on_end)

        t_run_start = time.monotonic()
        logger.info(
            f"trial_run_start instance={raw_id} sample_idx={sample_idx}",
            extra={
                "event": "trial_run_start",
                "instance_id": raw_id,
                "sample_idx": sample_idx,
            },
        )
        result = await trial.run()
        logger.info(
            f"trial_run_done instance={raw_id} sample_idx={sample_idx}",
            extra={
                "event": "trial_run_done",
                "instance_id": raw_id,
                "sample_idx": sample_idx,
                "duration_secs": round(time.monotonic() - t_run_start, 3),
            },
        )

        reward, eval_report = _extract_reward(result)
        exit_status = _extract_exit_status(result)
        agent_metrics = _extract_metrics(result)

        # -- Dashboard: finalize trial record --
        record.reward = reward
        record.exit_status = exit_status
        record.agent_metrics = agent_metrics
        record.end_time = time.time()
        record.phase = "done"
        record.trial_dir = str(Path(trial.trial_dir).resolve())
        _dashboard_state.active_trials.pop(record_key, None)
        _dashboard_state.trials.append(record)
        _dashboard_state._rollout_finished += 1
        _dashboard_state._update_rollout_progress()

        # Save trajectory to persistent shared storage
        _save_trajectory_final(record)
        _persist_trial(record)

        # Clean up local trial directory to prevent disk exhaustion.
        # All important artifacts have been copied to persistent shared
        # storage by _save_trajectory_final above.
        if record.trial_dir:
            local_trial = Path(record.trial_dir)
            if local_trial.exists():
                try:
                    shutil.rmtree(local_trial)
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up local trial dir {local_trial}: {e}",
                        extra={
                            "event": "cleanup_failed",
                            "instance_id": raw_id,
                            "sample_idx": sample_idx,
                            "trial_dir": str(local_trial),
                        },
                    )

        _dashboard_state._notify_subscribers(
            {
                "type": "trial_finished",
                "key": record_key,
                "instance_id": raw_id,
                "reward": reward,
                "exit_status": exit_status,
            }
        )

        return {
            "reward": reward,
            "exit_status": exit_status,
            "agent_metrics": agent_metrics,
            "eval_report": eval_report,
        }

    except asyncio.CancelledError:
        logger.info(
            "Harbor trial cancelled",
            extra={
                "event": "trial_cancelled",
                "instance_id": locals().get("raw_id", ""),
                "sample_idx": locals().get("sample_idx", -1),
            },
        )
        _mark_trial_cancelled(
            record_key=record_key,
            record=record,
            instance_id=locals().get("raw_id", ""),
            sample_idx=locals().get("sample_idx", -1),
            reason="Trial cancelled",
            final_save=True,
            cleanup_local=True,
        )
        raise

    except Exception as e:
        # If the Harbor exception carries a structured error_code (e.g.
        # downstream consumers can group by error_code without parsing
        # the message.
        _err_code = getattr(e, "code", None) or getattr(e, "error_code", None)
        logger.error(
            f"Harbor trial failed: {e}\n{traceback.format_exc()}",
            extra={
                "event": "trial_failed",
                "instance_id": locals().get("raw_id", ""),
                "sample_idx": locals().get("sample_idx", -1),
                "error_class": type(e).__name__,
                "error_code": _err_code if _err_code else "",
            },
        )
        # -- Dashboard: mark trial as error --
        if record_key and record_key in _dashboard_state.active_trials:
            record.phase = "error"
            record.error_msg = str(e)[:2000]
            record.end_time = time.time()
            _dashboard_state.active_trials.pop(record_key, None)
            _dashboard_state.trials.append(record)
            _dashboard_state._rollout_finished += 1
            _dashboard_state._update_rollout_progress()
            _dashboard_state.errors.append(
                {
                    "time": time.time(),
                    "msg": f"Trial {raw_id}#{sample_idx}: {str(e)[:2000]}",
                    "instance_id": raw_id,
                    "sample_idx": sample_idx,
                }
            )
            _dashboard_state.error_count += 1
            _persist_trial(record)
            _dashboard_state._notify_subscribers(
                {
                    "type": "trial_error",
                    "key": record_key,
                    "instance_id": raw_id,
                    "error": str(e)[:2000],
                }
            )
            # Clean up local trial dir on error too
            if record and record.trial_dir:
                try:
                    shutil.rmtree(Path(record.trial_dir))
                except Exception as _cleanup_e:
                    logger.warning(
                        f"Failed to clean up trial dir after error: {_cleanup_e}",
                        extra={
                            "event": "cleanup_after_error_failed",
                            "instance_id": locals().get("raw_id", ""),
                            "sample_idx": locals().get("sample_idx", -1),
                            "trial_dir": record.trial_dir,
                        },
                    )
        return _error_response(f"Error: {type(e).__name__}")


def get_semaphore() -> asyncio.Semaphore:
    """Per-worker concurrency guard, initialized in _lifespan. Always present
    once the server has started; the assert documents that invariant for the
    type checker (the module-level default is None before startup)."""
    assert _semaphore is not None, "Semaphore not initialized — server not started?"
    return _semaphore


@app.post("/run")
async def run_instance(http_request: Request, request: RunRequest) -> RunResponse:
    """Run an agent on a single task instance via Harbor."""
    global _run_inflight, _active_run_task_counter
    t_recv = time.monotonic()
    logger.info(
        f"Running instance: {request.instance_id}",
        extra={
            "event": "run_received",
            "instance_id": request.instance_id,
            "inflight_before_acquire": _run_inflight,
        },
    )

    sem = get_semaphore()
    await sem.acquire()

    result: dict[str, Any] = {
        "exit_status": "Unknown",
        "reward": 0.0,
        "agent_metrics": {},
        "eval_report": {},
    }
    run_task_id: int | None = None
    trial_task: asyncio.Task[dict[str, Any]] | None = None

    try:
        _run_inflight += 1
        t_acq = time.monotonic()
        wait_secs = round(t_acq - t_recv, 3)
        logger.info(
            f"run_acquired instance={request.instance_id} wait={wait_secs}s inflight={_run_inflight}",
            extra={
                "event": "run_acquired",
                "instance_id": request.instance_id,
                "wait_secs": wait_secs,
                "inflight_after_acquire": _run_inflight,
            },
        )

        _active_run_task_counter += 1
        run_task_id = _active_run_task_counter
        trial_task = asyncio.create_task(_run_trial(request))
        _active_run_tasks[run_task_id] = {
            "task": trial_task,
            "instance_id": request.instance_id,
            "started_at": time.time(),
            "session_base_url": _parse_session_base_url(request.base_url),
        }

        while True:
            done, _ = await asyncio.wait({trial_task}, timeout=1.0)
            if trial_task in done:
                try:
                    result = await trial_task
                except asyncio.CancelledError:
                    result = _error_response("Cancelled")
                break

            if await http_request.is_disconnected():
                logger.info(
                    f"run_client_disconnected instance={request.instance_id}; cancelling Harbor trial",
                    extra={
                        "event": "run_client_disconnected",
                        "instance_id": request.instance_id,
                    },
                )
                trial_task.cancel()
                try:
                    await trial_task
                except asyncio.CancelledError:
                    pass
                result = _error_response("Cancelled")
                break

    except asyncio.CancelledError:
        if trial_task is not None and not trial_task.done():
            logger.info(
                f"run_cancelled instance={request.instance_id}; cancelling Harbor trial",
                extra={
                    "event": "run_cancelled",
                    "instance_id": request.instance_id,
                },
            )
            trial_task.cancel()
            try:
                await trial_task
            except asyncio.CancelledError:
                pass
        result = _error_response("Cancelled")
        raise

    finally:
        if run_task_id is not None:
            _active_run_tasks.pop(run_task_id, None)
        _run_inflight -= 1
        sem.release()
        total_secs = round(time.monotonic() - t_recv, 3)
        logger.info(
            f"Instance {request.instance_id} finished: exit_status={result['exit_status']}, reward={result['reward']}",
            extra={
                "event": "run_released",
                "instance_id": request.instance_id,
                "exit_status": result.get("exit_status", ""),
                "reward": result.get("reward", 0.0),
                "duration_secs": total_secs,
                "inflight_after_acquire": _run_inflight,
            },
        )

    return RunResponse(**result)


@app.post("/abort_all")
async def abort_all() -> dict[str, Any]:
    """Abort all active Harbor trials in this worker process."""
    return await _abort_local_active_trials("abort_all")


@app.get("/health")
async def health():
    return {"status": "ok"}


# -- Dashboard endpoints --


@app.get("/api/dashboard/state")
async def dashboard_state():
    """Full dashboard state for polling."""
    return _dashboard_state.to_dict()


@app.get("/api/dashboard/events")
async def dashboard_events():
    """SSE stream for live dashboard updates."""

    async def event_stream():
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _dashboard_state.subscribers.append(q)
        try:
            yield "retry: 3000\n\n"
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _dashboard_state.subscribers.remove(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/dashboard/trials")
async def dashboard_trials(
    phase: str | None = Query(None),
    instance_id: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated trial list (active first, then completed)."""
    all_trials = []
    for r in _dashboard_state.active_trials.values():
        all_trials.append(r.to_dict())
    for r in reversed(_dashboard_state.trials):
        all_trials.append(r.to_dict())

    if phase:
        all_trials = [t for t in all_trials if t["phase"] == phase]
    if instance_id:
        all_trials = [t for t in all_trials if t["instance_id"] == instance_id]

    return {
        "total": len(all_trials),
        "offset": offset,
        "trials": all_trials[offset : offset + limit],
    }


@app.get("/api/dashboard/time-breakdown")
async def dashboard_time_breakdown():
    """Aggregated timing breakdown across completed trials."""
    import statistics

    env_setup_times = []
    agent_run_times = []
    llm_times = []
    tool_times = []
    eval_times = []
    total_times = []

    for r in _dashboard_state.trials:
        if r.phase != "done":
            continue
        am = r.agent_metrics or {}
        duration = r.end_time - r.start_time if r.end_time and r.start_time else 0

        # Phase-level timing
        env_setup = r.env_setup_sec or 0
        env_setup_times.append(env_setup)

        agent_time = am.get("agent_run_time", 0) or 0
        agent_run_times.append(agent_time)

        eval_time = am.get("eval_time", 0) or 0
        eval_times.append(eval_time)

        total_times.append(duration)

        # Break down agent time into LLM vs tool execution
        api_times = am.get("api_request_times_msec", [])
        if api_times:
            llm_sec = sum(api_times) / 1000.0
            tool_sec = max(0, agent_time - llm_sec)
        else:
            # Estimate: use model_time_ratio if available
            llm_sec = agent_time * 0.6  # fallback estimate
            tool_sec = agent_time * 0.4
        llm_times.append(llm_sec)
        tool_times.append(tool_sec)

    def median_or_zero(lst):
        return statistics.median(lst) if lst else 0

    med_env = median_or_zero(env_setup_times)
    med_llm = median_or_zero(llm_times)
    med_tool = median_or_zero(tool_times)
    med_eval = median_or_zero(eval_times)
    med_total = median_or_zero(total_times)
    weight_update = _dashboard_state.metrics.get("last_weight_update_sec", 0) or 0

    # Compute percentages (of median total)
    agent_total = med_llm + med_tool
    denom = med_env + agent_total + med_eval + weight_update or 1

    return {
        "n_trials": len(total_times),
        "median": {
            "env_setup_sec": round(med_env, 1),
            "llm_inference_sec": round(med_llm, 1),
            "tool_execution_sec": round(med_tool, 1),
            "verification_sec": round(med_eval, 1),
            "weight_update_sec": round(weight_update, 1),
            "total_sec": round(med_total, 1),
        },
        "pct": {
            "env_setup": round(100 * med_env / denom, 1),
            "llm_inference": round(100 * med_llm / denom, 1),
            "tool_execution": round(100 * med_tool / denom, 1),
            "verification": round(100 * med_eval / denom, 1),
            "weight_update": round(100 * weight_update / denom, 1),
        },
    }


@app.get("/api/dashboard/trajectory/{instance_id}/{sample_idx}")
async def dashboard_trajectory(instance_id: str, sample_idx: int):
    """Serve the ATIF trajectory JSON for a completed trial."""
    record = _find_trial_record(instance_id, sample_idx)
    if not record or not record.trial_dir:
        return {"error": "Trial not found or no trial directory"}

    # Try ephemeral trial_dir first (live data), then persistent_dir
    for base in [record.trial_dir, record.persistent_dir]:
        if not base:
            continue
        traj_path = Path(base) / "agent" / "trajectory.json"
        if not traj_path.exists():
            traj_path = Path(base) / "trajectory.json"
        if traj_path.exists():
            return FileResponse(traj_path, media_type="application/json")

    # Fallback: try result.json
    result_path = Path(record.trial_dir) / "result.json"
    if result_path.exists():
        return FileResponse(result_path, media_type="application/json")

    return {"error": "Trajectory file not found", "trial_dir": record.trial_dir}


@app.get("/api/dashboard/trial-log/{instance_id}/{sample_idx}")
async def dashboard_trial_log(instance_id: str, sample_idx: int):
    """Serve the raw trial.log for a trial."""
    record = _find_trial_record(instance_id, sample_idx)
    if not record or not record.trial_dir:
        return {"error": "Trial not found or no trial directory"}

    log_path = Path(record.trial_dir) / "trial.log"
    if log_path.exists():
        content = log_path.read_text(errors="replace")
        return {"log": content[-50000:]}  # last 50KB

    return {"error": "Trial log not found", "trial_dir": record.trial_dir}


@app.get("/api/dashboard/recording/{instance_id}/{sample_idx}")
async def dashboard_recording(instance_id: str, sample_idx: int):
    """Serve the asciinema recording.cast for a trial."""
    record = _find_trial_record(instance_id, sample_idx)
    if not record:
        return {"error": "Trial not found"}

    for base in [record.trial_dir, record.persistent_dir]:
        if not base:
            continue
        cast_path = Path(base) / "agent" / "recording.cast"
        if not cast_path.exists():
            cast_path = Path(base) / "recording.cast"
        if cast_path.exists():
            return FileResponse(cast_path, media_type="text/plain")

    return {"error": "Recording not found"}


@app.get("/api/dashboard/detailed-scores/{instance_id}/{sample_idx}")
async def dashboard_detailed_scores(instance_id: str, sample_idx: int):
    """Serve the detailed verifier scores for a trial."""
    record = _find_trial_record(instance_id, sample_idx)
    if not record:
        return {"error": "Trial not found"}

    for base in [record.trial_dir, record.persistent_dir]:
        if not base:
            continue
        scores_path = Path(base) / "verifier" / "detailed_scores.json"
        if not scores_path.exists():
            scores_path = Path(base) / "detailed_scores.json"
        if scores_path.exists():
            return FileResponse(scores_path, media_type="application/json")

    return {"error": "Detailed scores not found"}


def _read_text_tail(
    path: Path, max_bytes: int = _MAX_VERIFIER_LOG_BYTES
) -> tuple[str, bool]:
    data = path.read_bytes()
    truncated = len(data) > max_bytes
    if truncated:
        data = data[-max_bytes:]
    return data.decode(errors="replace"), truncated


@app.get("/api/dashboard/verifier-output/{instance_id}/{sample_idx}")
async def dashboard_verifier_output(instance_id: str, sample_idx: int):
    """Serve verifier stdout/stderr and reward artifacts for a trial."""
    record = _find_trial_record(instance_id, sample_idx)
    if not record:
        return {"error": "Trial not found"}

    files: dict[str, dict[str, Any]] = {}
    for key, filename, label in _VERIFIER_OUTPUT_FILES:
        for base in [record.trial_dir, record.persistent_dir]:
            if not base:
                continue
            base_path = Path(base)
            candidates = [base_path / "verifier" / filename, base_path / filename]
            for path in candidates:
                if path.exists():
                    try:
                        content, truncated = _read_text_tail(path)
                    except FileNotFoundError:
                        continue
                    except OSError as e:
                        logger.warning(f"Failed to read verifier output {path}: {e}")
                        continue
                    files[key] = {
                        "filename": filename,
                        "label": label,
                        "content": content,
                        "truncated": truncated,
                    }
                    break
            if key in files:
                break

    if not files:
        return {"error": "Verifier output not found"}

    return {"files": files}


def _find_trial_record(instance_id: str, sample_idx: int) -> TrialRecord | None:
    """Find a trial record by instance_id and sample_idx."""
    key = f"{instance_id}_{sample_idx}"
    if key in _dashboard_state.active_trials:
        return _dashboard_state.active_trials[key]
    for r in _dashboard_state.trials:
        if r.instance_id == instance_id and r.sample_idx == sample_idx:
            return r
    return None


router_app = FastAPI(title="Agent Environment Server Router")
router_app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@router_app.get("/api/dashboard/state")
async def router_dashboard_state() -> dict[str, Any]:
    import httpx

    worker_ports = list(router_app.state.worker_ports)

    async def fetch_worker(port: int) -> dict[str, Any]:
        url = f"http://127.0.0.1:{port}/api/dashboard/state"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
            return {
                "worker_port": port,
                "ok": resp.status_code == 200,
                "state": resp.json() if resp.status_code == 200 else None,
                "status_code": resp.status_code,
            }
        except Exception as e:
            return {
                "worker_port": port,
                "ok": False,
                "state": None,
                "error": repr(e),
            }

    results = await asyncio.gather(*(fetch_worker(port) for port in worker_ports))

    active_trials = []
    recent_trials = []
    errors = []
    total = 0
    done = 0
    error_count = 0

    for item in results:
        state = item.get("state")
        if not item.get("ok") or not isinstance(state, dict):
            continue

        active_trials.extend(state.get("active_trials", []))
        recent_trials.extend(state.get("recent_trials", []))
        errors.extend(state.get("errors", []))

        rp = state.get("rollout_progress", {}) or {}
        total += int(rp.get("total", 0) or 0)
        done += int(rp.get("done", 0) or 0)

        error_count += int(state.get("error_count", 0) or 0)

    pct = int(100 * done / max(1, total))

    return {
        "pipeline_phase": "multiworker",
        "rollout_progress": {
            "done": done,
            "total": total,
            "pct": pct,
            "rollout_id": 0,
        },
        "training_step": 0,
        "metrics": {},
        "error_count": error_count,
        "last_error": errors[-1]["msg"] if errors else "",
        "errors": errors[-50:],
        "sglang_throughput": 0.0,
        "cluster_info": {
            "workers": len(worker_ports),
            "worker_ports": worker_ports,
        },
        "active_trials": active_trials,
        "recent_trials": recent_trials,
        "workers": results,
    }


@router_app.get("/health")
async def router_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "router",
        "worker_ports": router_app.state.worker_ports,
    }


@router_app.post("/abort_all")
async def router_abort_all() -> dict[str, Any]:
    import httpx

    worker_ports = list(router_app.state.worker_ports)

    async def call_worker(port: int) -> dict[str, Any]:
        url = f"http://127.0.0.1:{port}/abort_all"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url)
            body = resp.json()
            if isinstance(body, dict):
                body["worker_port"] = port
                body["status_code"] = resp.status_code
                return body
            return {
                "worker_port": port,
                "status_code": resp.status_code,
                "error": "non_dict_response",
            }
        except Exception as e:
            return {"worker_port": port, "error": repr(e)}

    results = await asyncio.gather(*(call_worker(port) for port in worker_ports))
    aborted_trials = sum(int(item.get("aborted_trials", 0)) for item in results)
    already_done = sum(int(item.get("already_done", 0)) for item in results)

    logger.warning(
        f"abort_all broadcast_http: workers={len(worker_ports)} "
        f"responses={len(results)} aborted_trials={aborted_trials} already_done={already_done}",
        extra={
            "event": "abort_all_broadcast_http",
            "workers": len(worker_ports),
            "responses": len(results),
            "aborted_trials": aborted_trials,
            "already_done": already_done,
        },
    )

    return {
        "status": "aborting",
        "mode": "router_http_broadcast",
        "workers": len(worker_ports),
        "responses": len(results),
        "aborted_trials": aborted_trials,
        "already_done": already_done,
        "worker_results": results,
    }


async def _proxy_to_worker(worker_idx: int, path: str, request: Request):
    import httpx

    port = router_app.state.worker_ports[worker_idx]
    url = f"http://127.0.0.1:{port}/{path}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "connection"}
    }

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(
            request.method,
            url,
            content=body,
            headers=headers,
            params=request.query_params,
        )

    return StreamingResponse(
        iter([resp.content]),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


@router_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def router_round_robin(path: str, request: Request):
    # Round-robin all non-/abort_all requests across worker partitions.
    worker_idx = router_app.state.rr_counter % len(router_app.state.worker_ports)
    router_app.state.rr_counter += 1
    return await _proxy_to_worker(worker_idx, path, request)


def _run_worker_process(
    worker_id: int,
    host: str,
    port: int,
    max_concurrent: int,
    num_workers: int,
) -> None:
    global _WORKER_ID
    _WORKER_ID = worker_id

    os.environ["AGENT_MAX_CONCURRENT"] = str(max_concurrent)
    os.environ["HARBOR_NUM_WORKERS"] = str(num_workers)
    os.environ["HARBOR_WORKER_ID"] = str(worker_id)
    os.environ.setdefault("OPENAI_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)
    os.environ.setdefault("MSWEA_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)
    os.environ.setdefault("HOSTED_VLLM_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)

    install_json_formatter()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    uvicorn.run(app, host=host, port=port, workers=1)


def _run_router_with_workers(args) -> None:
    worker_ports = [args.worker_base_port + i for i in range(args.workers)]

    processes = []
    for worker_id, worker_port in enumerate(worker_ports):
        proc = mp.Process(
            target=_run_worker_process,
            args=(
                worker_id,
                args.host,
                worker_port,
                args.max_concurrent,
                args.workers,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    router_app.state.worker_ports = worker_ports
    router_app.state.rr_counter = 0

    try:
        uvicorn.run(
            router_app,
            host=args.host,
            port=args.port,
            workers=1,
            log_config=_harbor_log_config(args.log_level),
        )
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)


def _harbor_log_config(level: str) -> dict[str, Any]:
    """uvicorn log_config that sets Harbor's shared logger level.

    uvicorn pickles its Config (including log_config) into every spawned
    worker and runs logging.config.dictConfig() there, so this is the
    env-var-free way to make all --workers processes agree on Harbor's level.
    `incremental` adjusts only the level and leaves the handlers/formatters
    owned by install_json_formatter untouched. Harbor's setup_logger guards
    its own setLevel, so its lazy import does not override this.
    """
    return {
        "version": 1,
        "incremental": True,
        "loggers": {"harbor.utils.logger": {"level": level}},
    }


def main():
    parser = argparse.ArgumentParser(description="Agent Environment Server (Harbor)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    # Per-worker ceiling for the always-on safety semaphore (see _lifespan).
    # Defaults high so it is a runaway guard, not a routine cap — the server-wide
    # ceiling is workers * max_concurrent.
    parser.add_argument("--max-concurrent", type=int, default=40000)
    # Each worker is its own process + event loop. Spreading rollout load
    # across workers stops one saturated event loop from starving the whole
    # server (single-loop saturation -> heartbeat death -> crash).
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-base-port", type=int, default=12000)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Level for Harbor's shared logger (harbor.utils.logger) and its children.",
    )
    args = parser.parse_args()

    # The levels are exported into the environment because their consumers read
    # them at import time: harbor's setup_logger reads HARBOR_LOG_LEVEL, LiteLLM
    # reads LITELLM_LOG, and install_json_formatter reads LOG_LEVEL for the root
    # handler. Worker processes inherit this environment, so the levels reach
    # every worker without being re-passed through _run_worker_process.
    os.environ["HARBOR_LOG_LEVEL"] = args.log_level.upper()

    os.environ["AGENT_MAX_CONCURRENT"] = str(args.max_concurrent)
    os.environ["HARBOR_NUM_WORKERS"] = str(args.workers)

    os.environ.setdefault("OPENAI_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)
    os.environ.setdefault("MSWEA_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)
    os.environ.setdefault("HOSTED_VLLM_API_KEY", _OPENAI_API_KEY_PLACEHOLDER)

    install_json_formatter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log_config = _harbor_log_config(args.log_level)
    if args.workers > 1:
        _run_router_with_workers(args)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_config=log_config)


if __name__ == "__main__":
    main()
