# Modified for release: configurable deployment paths and public source references.
"""JSON log formatter for harbor_server and adjacent modules.

Single-line JSON records, one per log line. Keys are stable so downstream
consumers (jq, grep, external log pipelines) don't depend on free-form
message text.

Top-level schema (every record):
  ts            ISO-8601 UTC with milliseconds, e.g. "2026-05-28T20:14:38.123Z"
  level         "INFO" | "WARNING" | "ERROR" | "DEBUG" | "CRITICAL"
  logger        Python logger name, e.g. "agent360.harbor.miles.harbor_server"
  message       The rendered log message string
  slurm_job_id  From env SLURM_JOB_ID at process start (cached)
  pid           Process id (handy under uvicorn multi-worker)
  event         Optional structured event tag (from extra={"event": ...})

Additional per-event fields flow through via `logger.info(msg, extra={...})`.
A whitelist controls which `extra` keys are emitted (so unrelated dict keys
present in LogRecord don't leak).

Exception records add:
  exc_type, exc_message, exc_traceback
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any

# Sanitize SLURM_JOB_ID so it's safe as an identifier in downstream log
# pipelines (path components, partition keys, grep filters). Replace anything
# outside [A-Za-z0-9._-] with `-`; default to `local-dev` when unset so local
# runs are queryable as a single distinct slurm_job_id value.
_SAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_slurm_job_id(raw: str) -> str:
    if not raw:
        return "local-dev"
    return _SAFE_ID_CHARS.sub("-", raw)


# Cached at module import so per-record formatting stays cheap.
_SLURM_JOB_ID = _sanitize_slurm_job_id(os.environ.get("SLURM_JOB_ID", ""))
_PID = os.getpid()

# ============================================================================
# WHITELIST of `extra=` keys that get promoted to top-level JSON record fields.
# ============================================================================
# Keeping the whitelist explicit prevents accidental dict keys (record.args,
# record.msg, random typos) from leaking into the wire format. Anything passed
# in `extra={...}` that is NOT in this list is silently dropped at format time.
#
# How `message` works (NOT controlled by this whitelist):
#   logger.info("trial_run_done instance=smoke-1", extra={"event": "trial_run_done", ...})
# yields  message="trial_run_done instance=smoke-1"  AND  event="trial_run_done"
# as separate top-level JSON fields. The first positional arg is `message`;
# extras are separate top-level fields. They never get interpolated together.
#
# To add a new field:
#   1. Append the field name below
#   2. Update the field descriptions below
#
# Field                       | Emitted by                          | Example
# ----------------------------+-------------------------------------+-------------------------------
# event                       | every structured emission           | "run_acquired"
# instance_id                 | every per-trial event               | "smoke-1"
# sample_idx                  | every per-trial event               | 7
# rollout_id                  | reserved — not yet emitted          | —
# wait_secs                   | run_acquired                        | 0.234
# inflight_before_acquire     | run_received                        | 12
# inflight_after_acquire      | run_acquired, run_released          | 13
# max_concurrent              | semaphore_initialized               | 8
# duration_secs               | trial_create_done, trial_run_done,  | 0.342
#                             | run_released                        |
# phase                       | each phase hook                     | "env_setup"
# phase_duration_sec          | phase hooks (except env_setup)      | 12.3
# env_setup_sec               | phase_agent_running                 | 8.1
# trial_dir                   | trial_create_done                   | "./outputs/trial-abc"
# error_code                  | trial_failed, error paths           | "CLUSTER_FULL"
# error_class                 | trial_failed                        | "ValueError"
# exit_status                 | run_released                        | "Submitted"
# reward                      | run_released                        | 1.0
# subscribers_count           | dashboard_queue_full                | 5
# dropped_count               | dashboard_queue_full                | 127
# endpoint                    | reserved — viewer post failures     | —
# task_path                   | task_not_found                      | "/root/harbor_tasks/foo"
_ALLOWED_EXTRA_FIELDS: frozenset[str] = frozenset({
    "event",
    "instance_id",
    "sample_idx",
    "rollout_id",
    "wait_secs",
    "inflight_after_acquire",
    "inflight_before_acquire",
    "max_concurrent",
    "duration_secs",
    "phase",
    "phase_duration_sec",
    "env_setup_sec",
    "trial_dir",
    "error_code",
    "error_class",
    "exit_status",
    "reward",
    "subscribers_count",
    "dropped_count",
    "endpoint",
    "task_path",
})


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record. Stable schema for downstream consumers."""

    def format(self, record: logging.LogRecord) -> str:
        # ts: use record.created (seconds since epoch) so the timestamp matches
        # when the event happened, not when the formatter ran.
        ts_secs, ts_ms = divmod(record.created * 1000.0, 1000.0)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts_secs)) + (".%03dZ" % int(ts_ms))

        obj: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "slurm_job_id": _SLURM_JOB_ID,
            "pid": _PID,
        }

        # Pull whitelisted extras off the LogRecord.
        for k in _ALLOWED_EXTRA_FIELDS:
            v = getattr(record, k, None)
            if v is not None:
                obj[k] = v

        # Exception info, if present.
        if record.exc_info:
            etype, evalue, _tb = record.exc_info
            obj["exc_type"] = etype.__name__ if etype else ""
            obj["exc_message"] = str(evalue) if evalue else ""
            obj["exc_traceback"] = self.formatException(record.exc_info)

        # default=str so unexpected non-JSON-serializable values don't kill the
        # logger. Better to ship a slightly-wrong record than to drop it.
        return json.dumps(obj, default=str, ensure_ascii=False)


def install_json_formatter(stream=None) -> logging.StreamHandler:
    """Idempotently install JSON formatting on the root logger.

    Replaces any existing handlers on the root logger with a single
    StreamHandler emitting JSON. Safe to call multiple times — each call
    is a no-op if a JSON-formatted StreamHandler is already attached to
    the requested stream.

    Returns the StreamHandler installed (or already present).

    The default stream is sys.stderr so log output continues to flow
    through the existing `> harbor.log 2>&1` redirect in
    `launch_harbor_server.sh:106` without further changes.
    """
    if stream is None:
        stream = sys.stderr

    root = logging.getLogger()

    # Idempotency: if a JSON-formatter handler is already attached to this
    # stream, do nothing. Identifying it by the formatter type avoids
    # repeated re-installation under uvicorn worker re-imports + lifespan
    # double-init.
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is stream:
            if isinstance(h.formatter, JsonFormatter):
                return h

    # Remove any pre-existing handlers on the root logger to avoid duplicate
    # output (text + JSON). Caller can re-add their own afterwards if needed.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    # Honor LOG_LEVEL if set, else INFO. basicConfig used to do the equivalent.
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return handler
