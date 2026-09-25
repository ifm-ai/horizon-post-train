"""
Custom agent function for agentic_tool_call.generate.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).

This implementation is self-contained on the HTTP client side:
- no dependency on miles.utils.http_utils.post
- one AsyncClient per process/event-loop
- bounded concurrent POSTs
- no global-client replacement leak when event loops change
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import weakref
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from agent360.harbor.miles.agent_config import (
    MINI_SWE_AGENT_NAME,
    AgentConfigError,
    deep_merge_dicts,
    validate_mini_swe_agent_config,
)
from agent360.harbor.miles.utils import coerce_reward_to_float

logger = logging.getLogger(__name__)

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT_SECS", "3600"))
REQUIRED_SAMPLING_PARAMS = ("max_tokens", "temperature", "top_p")

# Client-side concurrency. Set this to match or exceed the Harbor agent server's
# intended request concurrency. Default is intentionally high enough to avoid
# becoming the bottleneck in rollout-heavy jobs.
AGENT_CLIENT_MAX_CONNECTIONS = int(os.getenv("AGENT_CLIENT_MAX_CONNECTIONS", "1024"))
AGENT_CLIENT_MAX_KEEPALIVE = int(
    os.getenv(
        "AGENT_CLIENT_MAX_KEEPALIVE",
        str(min(AGENT_CLIENT_MAX_CONNECTIONS, 256)),
    )
)
AGENT_CONNECT_TIMEOUT_SECS = float(os.getenv("AGENT_CONNECT_TIMEOUT_SECS", "600"))
AGENT_RUN_MAX_RETRIES = int(os.getenv("AGENT_RUN_MAX_RETRIES", "2"))

_RETRYABLE_POST_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

_RETRYABLE_POST_STATUSES = {
    429,
    502,
    503,
    504,
}

# One client per event loop. Do not use a single global AsyncClient across
# unrelated loops; httpx.AsyncClient is loop-affine in practice because its
# underlying async transport is tied to the async backend.
#
# WeakKeyDictionary prevents the loop key itself from being kept alive by this
# registry. Explicit cleanup is still better; see close_http_clients().
_ClientState = tuple[httpx.AsyncClient, asyncio.Semaphore]
_http_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _ClientState] = (
    weakref.WeakKeyDictionary()
)
_http_clients_lock = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _resolve_request_agent_config(
    agent_config: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if agent_config is None:
        run_config = {}
    elif isinstance(agent_config, dict):
        run_config = agent_config
    else:
        raise ValueError("agent_config must be a mapping.")

    metadata_config = metadata.get("agent_config")
    if metadata_config is None:
        return run_config
    if not isinstance(metadata_config, dict):
        raise ValueError("metadata['agent_config'] must be a mapping.")
    return deep_merge_dicts(run_config, metadata_config)


def _resolve_agent_name(metadata: dict[str, Any]) -> str:
    # Agent selection is metadata-driven: each task's metadata["agent_name"]
    # is the source of truth. AGENT_IMPL_OVERRIDE is a dev/testing escape
    # hatch — when set it forces every trial to use that agent regardless of
    # metadata. Leave it unset in production.
    override = os.getenv("AGENT_IMPL_OVERRIDE")
    if override:
        metadata_agent = metadata.get("agent_name")
        if metadata_agent and metadata_agent != override:
            logger.warning(
                "AGENT_IMPL_OVERRIDE=%s set; overriding task metadata agent_name=%s.",
                override,
                metadata_agent,
            )
        return override

    metadata_agent = metadata.get("agent_name")
    if metadata_agent:
        return metadata_agent

    raise ValueError(
        "Agent implementation is not configured. Set task "
        "metadata['agent_name'] or AGENT_IMPL_OVERRIDE."
    )


def _get_llm_timeout_secs() -> float:
    """Return the downstream Harbor LLM HTTP timeout in seconds."""
    raw_timeout = os.getenv("AGENT_LLM_TIMEOUT_SECS")
    if raw_timeout is None:
        raise ValueError(
            "Miles Harbor rollouts require AGENT_LLM_TIMEOUT_SECS to be set."
        )

    try:
        timeout = float(raw_timeout)
    except ValueError:
        raise ValueError(
            f"AGENT_LLM_TIMEOUT_SECS must be a positive number, got {raw_timeout!r}."
        ) from None

    if timeout <= 0:
        raise ValueError(f"AGENT_LLM_TIMEOUT_SECS must be > 0, got {raw_timeout!r}.")

    return timeout


def _get_agent_model_name() -> str:
    model_name = os.getenv("AGENT_MODEL_NAME")
    if not model_name:
        raise ValueError("Miles Harbor rollouts require AGENT_MODEL_NAME to be set.")
    return model_name


def _require_sampling_params(request_kwargs: dict[str, Any]) -> None:
    missing = [
        key for key in REQUIRED_SAMPLING_PARAMS if request_kwargs.get(key) is None
    ]
    if missing:
        raise ValueError(
            "Miles Harbor rollouts require explicit sampling params: "
            f"{', '.join(missing)}"
        )


def _get_http_client() -> _ClientState:
    """Return a process-local/event-loop-local AsyncClient and semaphore.

    This avoids both failure modes from the old pattern:
    1. client-side bottleneck from a too-small shared connection pool
    2. leaked clients from overwriting one global AsyncClient when loops change
    """
    loop = asyncio.get_running_loop()

    with _http_clients_lock:
        existing = _http_clients.get(loop)
        if existing is not None:
            client, semaphore = existing
            if not client.is_closed:
                return client, semaphore

        limits = httpx.Limits(
            max_connections=AGENT_CLIENT_MAX_CONNECTIONS,
            max_keepalive_connections=AGENT_CLIENT_MAX_KEEPALIVE,
            keepalive_expiry=30.0,
        )

        # Match the old behavior for long-running agent calls: no read/write/pool
        # timeout. Keep only a finite connect timeout so bad endpoints fail.
        timeout = httpx.Timeout(
            connect=AGENT_CONNECT_TIMEOUT_SECS,
            read=None,
            write=None,
            pool=None,
        )

        client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            headers={"Connection": "keep-alive"},
        )
        semaphore = asyncio.Semaphore(AGENT_CLIENT_MAX_CONNECTIONS)
        _http_clients[loop] = (client, semaphore)

        logger.info(
            "Initialized Harbor agent HTTP client: max_connections=%s "
            "max_keepalive=%s connect_timeout=%s run_max_retries=%s",
            AGENT_CLIENT_MAX_CONNECTIONS,
            AGENT_CLIENT_MAX_KEEPALIVE,
            AGENT_CONNECT_TIMEOUT_SECS,
            AGENT_RUN_MAX_RETRIES,
        )

        return client, semaphore


async def close_http_clients() -> None:
    """Explicitly close all HTTP clients owned by this module.

    This is optional for the dynamically loaded generate path, but useful if the
    worker process has a clean shutdown hook.
    """
    with _http_clients_lock:
        clients = [client for client, _ in _http_clients.values()]
        _http_clients.clear()

    await asyncio.gather(
        *(client.aclose() for client in clients if not client.is_closed),
        return_exceptions=True,
    )


async def _post_json(url: str, payload: dict[str, Any]) -> Any:
    """POST JSON with conservative retries for non-idempotent /run calls.

    Only retry failures that strongly suggest the request did not reach the
    application handler. Do not retry ambiguous in-flight failures such as
    ReadError, WriteError, RemoteProtocolError, ReadTimeout, or WriteTimeout.
    """
    client, semaphore = _get_http_client()

    last_exc: Exception | None = None

    for attempt in range(1, AGENT_RUN_MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post(url, json=payload)

            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError:
                    return response.text

            if (
                response.status_code in _RETRYABLE_POST_STATUSES
                and attempt < AGENT_RUN_MAX_RETRIES
            ):
                delay = min(2 ** (attempt - 1), 10) * (0.5 + random.random())

                logger.info(
                    "Agent server returned retryable HTTP %s "
                    "(attempt %s/%s, url=%s, delay=%.2fs, body=%s)",
                    response.status_code,
                    attempt,
                    AGENT_RUN_MAX_RETRIES,
                    url,
                    delay,
                    response.text[:500],
                )

                await asyncio.sleep(delay)
                continue

            response.raise_for_status()

        except _RETRYABLE_POST_EXCEPTIONS as exc:
            last_exc = exc

            if attempt >= AGENT_RUN_MAX_RETRIES:
                raise

            delay = min(2 ** (attempt - 1), 10) * (0.5 + random.random())

            logger.info(
                "Agent server POST failed before request was safely completed "
                "with %s (attempt %s/%s, url=%s, delay=%.2fs): %s",
                type(exc).__name__,
                attempt,
                AGENT_RUN_MAX_RETRIES,
                url,
                delay,
                exc,
            )

            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    agent_config: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    ).rstrip("/")
    model_name = _get_agent_model_name()

    # Always use the Miles session tracer URL (base_url) so the session
    # server can record the /v1/chat/completions request/response pair as
    # a SessionRecord. The session server's backend is the sgl-model-gateway
    # (wired via --sglang-router-ip/port), so going through the tracer still
    # exercises the gateway.
    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    agent_name = _resolve_agent_name(metadata)
    _require_sampling_params(request_kwargs)
    resolved_agent_config = _resolve_request_agent_config(agent_config, metadata)
    if resolved_agent_config and agent_name != MINI_SWE_AGENT_NAME:
        raise AgentConfigError(
            "agent_config is only supported for agent_name='mini-swe-agent-external'."
        )

    tool_format = os.getenv("AGENT_TOOL_FORMAT")
    llm_call_kwargs: dict[str, Any] = {
        "extra_body": {
            "return_completion_token_ids": True,
            "return_routed_experts": True,
            **({"chat_template_kwargs": {"tool_format": tool_format}} if tool_format else {}),
        }
    }

    max_tokens_cap = request_kwargs.get("max_tokens")
    if max_tokens_cap:
        llm_call_kwargs["max_tokens"] = int(max_tokens_cap)

    request = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
        "llm_call_kwargs": llm_call_kwargs,
        "llm_timeout_sec": _get_llm_timeout_secs(),
        "agent_name": agent_name,
    }
    if resolved_agent_config:
        request["agent_config"] = validate_mini_swe_agent_config(resolved_agent_config)

    max_seq_len = metadata.get("max_seq_len") or kwargs.get("max_seq_len")
    if max_seq_len:
        request["max_seq_len"] = max_seq_len

    logger.info(
        "Calling agent server: %s/run with instance_id=%s, base_url=%s",
        agent_server_url,
        metadata.get("instance_id", "?"),
        session_url,
    )

    try:
        response = await asyncio.wait_for(
            _post_json(f"{agent_server_url}/run", request),
            timeout=AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Agent server timed out after %ss for instance_id=%s",
            AGENT_TIMEOUT,
            metadata.get("instance_id", "?"),
        )
        return None
    except Exception as exc:
        logger.error("Agent server call failed: %s: %r", type(exc).__name__, exc)
        return None

    if not isinstance(response, dict):
        logger.error(
            "Agent server returned non-dict response for instance_id=%s: %r",
            metadata.get("instance_id", "?"),
            response,
        )
        return None

    logger.info(
        "Agent server response: instance_id=%s, reward=%s, exit_status=%s",
        metadata.get("instance_id", "?"),
        response.get("reward", "?"),
        response.get("exit_status", "?"),
    )

    return {
        "reward": coerce_reward_to_float(response.get("reward")),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }
