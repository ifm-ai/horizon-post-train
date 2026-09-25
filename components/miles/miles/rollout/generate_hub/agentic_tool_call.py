"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting one server-merged training sample

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      agent_config: dict,   # optional, only when --custom-agent-config-json is set
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
"""

import argparse
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import orjson
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    apply_merged_session_sample,
    truncate_samples_by_total_tokens,
)
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert getattr(input.args, "session_server_ip", None) and getattr(input.args, "session_server_port", None), (
        "agentic_tool_call.generate requires session_server_ip/session_server_port. "
        "Pass --use-session-server to start the session server."
    )
    capture = not input.evaluation and input.sampling_params["top_p"] < 1
    tracer = await OpenAIEndpointTracer.create(input.args, capture_sampling_mask=capture)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)
    assert not getattr(input.args, "generate_multi_samples", False), (
        "agentic_tool_call.generate with session-server merged collection produces one merged sample; "
        "--generate-multi-samples is not supported."
    )

    metadata = input.sample.metadata
    metadata = {**metadata, "sample_idx": input.sample.index}

    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    session_ip = getattr(input.args, "session_server_ip", None)
    session_port = getattr(input.args, "session_server_port", None)
    if session_ip and session_port:
        metadata = {**metadata, "session_server_id": f"{session_ip}:{session_port}"}

    agent_metadata = None
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            **build_agent_function_kwargs(
                input.args,
                base_url=tracer.base_url,
                prompt=input.sample.prompt,
                sampling_params=input.sampling_params,
                metadata=metadata,
            )
        )
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except Exception as e:
        logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)

    finally:
        logger.debug(f"{log_prefix} Calling collect_merged_sample...")
        merged_sample, session_metadata = await tracer.collect_merged_sample()
        if merged_sample is None:
            logger.debug(f"{log_prefix} collect_merged_sample done: empty sample")
        else:
            logger.debug(
                f"{log_prefix} collect_merged_sample done: tokens={len(merged_sample.tokens)} "
                f"response_length={merged_sample.response_length} "
                f"loss_mask={len(merged_sample.loss_mask)} "
                f"rollout_log_probs={len(merged_sample.rollout_log_probs)} "
                f"status={merged_sample.status}"
            )

    if merged_sample is None:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    if capture and merged_sample.rollout_sampling_mask is None:
        raise ValueError("a replay trajectory must carry a sampling mask")
    sample = apply_merged_session_sample(input.args, input.sample, merged_sample)
    sample.metadata.update(agent_metadata or {})
    sample.metadata.update(session_metadata)
    samples = [sample]

    if max_seq_len is not None:
        samples = truncate_samples_by_total_tokens(samples, max_seq_len, input.state.tokenizer)

    if not samples:
        logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    sample = samples[0]
    _mark_limits_exceeded_truncated(sample, agent_metadata)
    logger.debug(
        f"{log_prefix} server-merged sample ready: "
        f"tokens={len(sample.tokens)} response_length={sample.response_length} "
        f"total_time={time.monotonic()-t_start:.1f}s"
    )
    return GenerateFnOutput(samples=sample)


def _mark_limits_exceeded_truncated(
    samples: Sample | list[Sample],
    agent_metadata: dict[str, Any] | None,
) -> None:
    """Record turn-budget exhaustion even when the final model turn completed."""
    if (agent_metadata or {}).get("exit_status") == "LimitsExceeded":
        final_sample = samples if isinstance(samples, Sample) else samples[-1]
        final_sample.status = Sample.Status.TRUNCATED


def build_agent_function_kwargs(
    args: argparse.Namespace,
    *,
    base_url: str,
    prompt: Any,
    sampling_params: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    kwargs = {
        "base_url": base_url,
        "prompt": prompt,
        "request_kwargs": build_chat_request_kwargs(sampling_params),
        "metadata": metadata,
    }
    agent_config = getattr(args, "custom_agent_config", None)
    if agent_config is not None:
        kwargs["agent_config"] = agent_config
    return kwargs


def parse_custom_agent_config_json(value: str) -> dict[str, Any]:
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument(
        "--custom-agent-config-json",
        type=parse_custom_agent_config_json,
        default=None,
        dest="custom_agent_config",
        help="JSON object passed to the custom agent function as agent_config.",
    )
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) "
        "per session. Truncates samples on the Miles side and is forwarded to the "
        "Harbor agent server (as max_seq_len) to abort the trial early.",
    )


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
