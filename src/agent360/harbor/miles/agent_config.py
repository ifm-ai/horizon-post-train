"""Shared checks for MiniSweAgentExternal settings from RL360 YAML.

Users can put MiniSweAgentExternal settings in agent.yaml. Miles sends those
settings to swe_agent_function.run as agent_config, and that function posts them
to the Harbor server. This file keeps the client-side and server-side checks in
one place so they do not drift apart.

The allowed top-level keys are Python arguments passed when MiniSweAgentExternal
is created. Examples include step_limit, command_timeout_sec, agent_overrides,
model_overrides, and environment_overrides.

Some model connection and sampling values are set by the training stack, not by
agent.yaml. Examples include instance_id, api_base, api_key, max_tokens,
temperature, top_p, timeout, logprobs, and token-capture extra_body flags. Those
values are rejected loudly here so a bad config cannot silently change training
behavior.

Nested override maps stay open so new Harbor Mini-SWE options can be used
without changing RL360 every time.
"""

import math
from copy import deepcopy
from typing import Any


MINI_SWE_AGENT_NAME = "mini-swe-agent-external"

MINI_SWE_AGENT_CONFIG_KEYS = {
    "reasoning_effort",
    "cost_limit",
    "step_limit",
    "command_timeout_sec",
    "litellm_timeout_sec",
    "model_class",
    "agent_overrides",
    "model_overrides",
    "environment_overrides",
}
MINI_SWE_AGENT_MAPPING_KEYS = {
    "agent_overrides",
    "model_overrides",
    "environment_overrides",
}
PROTECTED_MODEL_KWARGS = {
    "api_base",
    "base_url",
    "api_key",
    "logprobs",
    "max_tokens",
    "temperature",
    "top_p",
    "timeout",
}
PROTECTED_MODEL_OVERRIDE_KEYS = {
    "instance_id",
}
PROTECTED_EXTRA_BODY_KEYS = {
    "return_token_ids",
    "return_prompt_token_ids",
    "return_completion_token_ids",
    "return_routed_experts",
}


class AgentConfigError(ValueError):
    """Invalid request-level agent configuration."""


def _require_config_str(config: dict[str, Any], key: str) -> None:
    if key not in config:
        return
    value = config[key]
    if not isinstance(value, str) or not value:
        raise AgentConfigError(f"agent_config.{key} must be a non-empty string.")


def _require_config_int(
    config: dict[str, Any],
    key: str,
    *,
    minimum: int,
) -> None:
    if key not in config:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AgentConfigError(
            f"agent_config.{key} must be an integer >= {minimum}."
        )


def _require_config_number(
    config: dict[str, Any],
    key: str,
    *,
    minimum: float,
    exclusive_minimum: bool = False,
) -> None:
    if key not in config:
        return
    value = config[key]
    valid_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    if exclusive_minimum:
        valid_number = valid_number and value > minimum
        comparator = ">"
    else:
        valid_number = valid_number and value >= minimum
        comparator = ">="
    if not valid_number:
        raise AgentConfigError(
            f"agent_config.{key} must be a finite number {comparator} {minimum}."
        )


def deep_merge_dicts(
    base: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    if not override:
        return deepcopy(base)

    merged = deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def validate_mini_swe_agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy after validating MiniSweAgentExternal kwargs."""
    if not isinstance(config, dict):
        raise AgentConfigError("agent_config must be a mapping.")

    unknown = sorted(set(config) - MINI_SWE_AGENT_CONFIG_KEYS)
    if unknown:
        raise AgentConfigError(
            "Unknown mini-swe-agent-external agent_config key(s): "
            + ", ".join(unknown)
        )

    for key in ("reasoning_effort", "model_class"):
        _require_config_str(config, key)
    _require_config_number(config, "cost_limit", minimum=0)
    _require_config_int(config, "step_limit", minimum=0)
    _require_config_int(config, "command_timeout_sec", minimum=1)
    _require_config_number(
        config,
        "litellm_timeout_sec",
        minimum=0,
        exclusive_minimum=True,
    )

    for key in MINI_SWE_AGENT_MAPPING_KEYS:
        value = config.get(key)
        if value is not None and not isinstance(value, dict):
            raise AgentConfigError(f"agent_config.{key} must be a mapping.")

    model_overrides = config.get("model_overrides") or {}
    protected_model_override_keys = sorted(
        set(model_overrides) & PROTECTED_MODEL_OVERRIDE_KEYS
    )
    if protected_model_override_keys:
        raise AgentConfigError(
            "agent_config.model_overrides cannot set training-controlled "
            "field(s): " + ", ".join(protected_model_override_keys)
        )

    model_kwargs = model_overrides.get("model_kwargs", {})
    if model_kwargs is None:
        model_kwargs = {}
    if "model_kwargs" in model_overrides and not isinstance(model_kwargs, dict):
        raise AgentConfigError(
            "agent_config.model_overrides.model_kwargs must be a mapping."
        )

    protected_model_keys = sorted(set(model_kwargs) & PROTECTED_MODEL_KWARGS)
    if protected_model_keys:
        raise AgentConfigError(
            "agent_config.model_overrides.model_kwargs cannot set RL-controlled "
            "field(s): " + ", ".join(protected_model_keys)
        )

    extra_body = model_kwargs.get("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if "extra_body" in model_kwargs and not isinstance(extra_body, dict):
        raise AgentConfigError(
            "agent_config.model_overrides.model_kwargs.extra_body must be a mapping."
        )
    protected_extra_body_keys = sorted(set(extra_body) & PROTECTED_EXTRA_BODY_KEYS)
    if protected_extra_body_keys:
        raise AgentConfigError(
            "agent_config.model_overrides.model_kwargs.extra_body cannot set "
            "RL-controlled field(s): " + ", ".join(protected_extra_body_keys)
        )

    return deepcopy(config)
