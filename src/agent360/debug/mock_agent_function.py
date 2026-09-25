# Modified for release: configurable deployment paths and public source references.
"""Harbor-bypass mock agent function for Miles agentic_tool_call.generate.

Matches the Miles custom-agent-function contract exactly (same signature as
agent360.harbor.miles.swe_agent_function.run) but skips the Harbor agent
server + Daytona sandbox entirely. Makes one or more real
/v1/chat/completions calls through the session-server tracer (which wraps
our sgl-model-gateway), returning a canned reward dict at the end.

What this DOES exercise:
  * Session server -> gateway -> engines request path
  * HiCache L1 / L2 / L3 (prompts are long enough to populate the cache)
  * PD disaggregation routing (prefill + decode workers, mooncake KV transport)
  * TITO: completion_token_ids in response (via return_completion_token_ids)
  * Routed experts capture (via return_routed_experts)
  * Multi-turn behavior (N turns configurable via env)
  * Per-rollout throughput measurement (emitted in agent_metrics)

What this does NOT exercise:
  * Harbor env setup / tool execution
  * Daytona sandbox provisioning
  * Real reward computation (reward is a fixed float)

Wire via:
  --custom-agent-function-path agent360.debug.mock_agent_function.run

Environment knobs (read at call time):
  MOCK_AGENT_TURNS        int      default 1   turns per rollout (2 caused concurrent
                                               chat pressure on GLM-4.7-Flash)
  MOCK_AGENT_REWARD       float    default 0.5 reward to emit
  MOCK_AGENT_TIMEOUT_SECS float    default 600 per /v1 call timeout (GLM-4.7-Flash
                                               cold-start + concurrent agent load can
                                               push a single call past 300s; 600s is
                                               generous so timeouts don't silently
                                               bin samples into the noop_filter drop
                                               path)
  MOCK_AGENT_MAX_TOKENS   int      default 32  max_tokens per turn (small so first
                                               inference is quick; goal is to exercise
                                               the path, not produce long responses)
  MOCK_AGENT_TEMPERATURE  float    default 0.7
  MOCK_AGENT_NONE_REWARD  "1"      force reward=None to exercise miles#989
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _default_messages(prompt: Any) -> list[dict[str, Any]]:
    """Build a minimal OpenAI chat request that exercises HiCache +
    PD disagg without needing a real task environment.

    The system prompt is deliberately long (~1000 chars) so HiCache has
    something to cache across turns and across concurrent rollouts.
    """
    system = (
        "You are a helpful coding assistant. When asked for code, respond "
        "with a short Python snippet. Follow best practices: use type hints, "
        "meaningful variable names, and explain tricky logic with a one-line "
        "comment. Do not include any markdown code fences in your response. "
        "Keep responses brief (under 200 tokens) so we can quickly iterate. "
        "If the user asks a follow-up, build on your previous answer rather "
        "than starting over. This context will repeat across many rollouts "
        "so the HiCache layer has a hot prefix to warm up with."
    )
    if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
        user_messages = prompt
    elif isinstance(prompt, str):
        user_messages = [{"role": "user", "content": prompt}]
    else:
        user_messages = [
            {
                "role": "user",
                "content": "Write a one-line Python function that returns the sum of two ints.",
            }
        ]
    return [{"role": "system", "content": system}, *user_messages]


async def _one_turn(
    chat_url: str,
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    timeout_secs: float,
) -> dict[str, Any] | None:
    payload = {
        "model": sampling_params.pop("model", "openai/fake-glm-4.7-flash"),
        "messages": messages,
        # SGLang accepts these as top-level ChatCompletionRequest fields.
        # They tell the engine to include TITO extras in the response; the
        # gateway forwards them through unchanged.
        "return_completion_token_ids": True,
        # return_routed_experts balloons response bodies (47 layers x top-k x
        # output tokens of expert indices), causing smg to drop connections
        # mid-stream. Not needed for FAST_ITER mock-agent path.
        "return_routed_experts": False,
        # Miles session tracer (miles/rollout/session/sessions.py) requires
        # meta_info.output_token_logprobs, completion_tokens, and prompt_token_ids
        # in the response, otherwise it returns 502 "meta_info and output_token_logprobs
        # must be in choice (requires logprobs=True)" and no SessionRecord is created.
        "logprobs": True,
        "return_prompt_token_ids": True,
        "return_meta_info": True,
        # Pull standard sampling params through.
        "temperature": sampling_params.get(
            "temperature", _env_float("MOCK_AGENT_TEMPERATURE", 0.7)
        ),
        "max_tokens": sampling_params.get(
            "max_tokens", _env_int("MOCK_AGENT_MAX_TOKENS", 32)
        ),
    }
    if "top_p" in sampling_params:
        payload["top_p"] = sampling_params["top_p"]

    # Low retry count because retrying /v1/chat/completions on the Miles
    # session-tracer is usually harmful: the first attempt already recorded
    # the assistant turn, so a retry sends stale messages that the tracer
    # rejects as 'rollback failed: no assistant message found in the first N
    # matched messages'. smg also has a known peer-closed bug (~4 bytes short)
    # on logprobs=True responses, which would trigger a retry cascade.
    # Capping retries limits the blast radius; failed trials still progress
    # to reward=0.0 (see all-turns-failed path below).
    max_retries = _env_int("MOCK_AGENT_MAX_RETRIES", 3)
    t0 = time.time()
    try:
        response = await asyncio.wait_for(
            post(chat_url, payload, action="post", max_retries=max_retries),
            timeout=timeout_secs,
        )
    except asyncio.TimeoutError:
        logger.error("mock_agent_function: /v1/chat/completions timed out after %.1fs", timeout_secs)
        return None
    except Exception as e:
        logger.error("mock_agent_function: /v1/chat/completions failed: %s", e)
        return None
    dt = time.time() - t0

    if not response or "choices" not in response:
        logger.warning("mock_agent_function: empty or malformed response: %r", response)
        return None

    choice = response["choices"][0]
    message = choice.get("message") or {}
    completion_token_ids = choice.get("completion_token_ids")
    routed_experts = choice.get("routed_experts")
    usage = response.get("usage") or {}

    n_completion_tokens = (
        len(completion_token_ids)
        if completion_token_ids is not None
        else usage.get("completion_tokens", 0)
    )
    tok_per_s = n_completion_tokens / dt if dt > 0 else 0.0

    return {
        "dt": dt,
        "n_tokens": n_completion_tokens,
        "tok_per_s": tok_per_s,
        "has_completion_token_ids": completion_token_ids is not None,
        "has_routed_experts": routed_experts is not None,
        "assistant_content": message.get("content") or "",
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Harbor-bypass mock agent. Exercises full inference stack only."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    # base_url is the tracer URL `{session_server}/sessions/{session_id}`.
    # Tracer proxies /v1/chat/completions through to the gateway while
    # recording the request/response pair as a SessionRecord. We just need
    # to hit the tracer's chat completions endpoint.
    chat_url = f"{base_url}/v1/chat/completions"
    # smg's pd_router filters workers by exact model_id match (no prefix fallback
    # beyond the literal UNKNOWN_MODEL_ID sentinel), so the request's "model"
    # field must match what engines register with. Engines register with the
    # full HF checkpoint path; prefer that, falling back to the agent-name envs.
    model_name = os.getenv(
        "AGENT_MODEL_FULL_PATH",
        os.getenv(
            "HF_MODEL_PATH",
            os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "fake-glm-4.7-flash")),
        ),
    )

    n_turns = max(1, _env_int("MOCK_AGENT_TURNS", 1))
    timeout_secs = _env_float("MOCK_AGENT_TIMEOUT_SECS", 600.0)

    messages = _default_messages(prompt)
    turn_metrics = []

    for turn in range(n_turns):
        # Miles session tracer requires logprobs + meta_info + prompt_token_ids
        # in the response (see miles/rollout/session/sessions.py:209). Without
        # these flags, the tracer's proxy validation rejects the response with
        # 502 and no SessionRecord is created, causing "No model calls recorded
        # for sample" downstream. We inject them here so mock agent requests
        # flow through the tracer cleanly and get recorded.
        # For HF checkpoint paths, pass through without the "openai/" prefix so
        # the request's model field matches engine-registered model_id exactly
        # (required by smg pd_router strict-match selection).
        model_field = model_name if model_name.startswith("/") else f"openai/{model_name}"
        sampling_params = {
            **request_kwargs,
            "model": model_field,
            "logprobs": True,
            "return_prompt_token_ids": True,
            "return_meta_info": True,
        }
        turn_result = await _one_turn(
            chat_url=chat_url,
            messages=messages,
            sampling_params=sampling_params,
            timeout_secs=timeout_secs,
        )
        if turn_result is None:
            logger.warning("mock_agent_function: turn %d failed, ending rollout", turn)
            break
        turn_metrics.append(turn_result)

        # Append assistant + a trivial user follow-up to exercise multi-turn
        # behavior (cache reuse, growing KV context).
        messages.append(
            {"role": "assistant", "content": turn_result["assistant_content"]}
        )
        if turn + 1 < n_turns:
            messages.append(
                {
                    "role": "user",
                    "content": "Can you add a short docstring and one example?",
                }
            )

    if not turn_metrics:
        # All turns failed. Return reward=0.0 (not None) because
        # miles/ray/rollout.py:650 _post_process_rewards crashes on None
        # with "TypeError: must be real number, not NoneType". miles#989
        # guards round(None) but not the torch.tensor conversion. Until
        # upstream handles None here, default failed trials to 0.
        return {
            "reward": 0.0,
            "exit_status": "LimitsExceeded",
            "eval_report": {},
            "agent_metrics": {"mock_agent/all_turns_failed": 1},
        }

    total_tokens = sum(t["n_tokens"] for t in turn_metrics)
    total_time = sum(t["dt"] for t in turn_metrics)
    median_tok_per_s = sorted(t["tok_per_s"] for t in turn_metrics)[len(turn_metrics) // 2]

    agent_metrics = {
        "mock_agent/n_turns": len(turn_metrics),
        "mock_agent/total_tokens": total_tokens,
        "mock_agent/total_time_s": total_time,
        "mock_agent/median_tok_per_s": median_tok_per_s,
        "mock_agent/had_completion_token_ids": int(
            any(t["has_completion_token_ids"] for t in turn_metrics)
        ),
        "mock_agent/had_routed_experts": int(
            any(t["has_routed_experts"] for t in turn_metrics)
        ),
    }

    reward: float | None = _env_float("MOCK_AGENT_REWARD", 0.5)
    if os.getenv("MOCK_AGENT_NONE_REWARD") == "1":
        reward = None

    exit_status = "Completed" if reward is not None else "LimitsExceeded"

    return {
        "reward": reward,
        "exit_status": exit_status,
        "eval_report": {
            "mock": True,
            "instance_id": metadata.get("instance_id", "mock"),
        },
        "agent_metrics": agent_metrics,
    }
