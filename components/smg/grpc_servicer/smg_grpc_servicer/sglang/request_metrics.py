"""Metric helpers for SGLang gRPC generation requests."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from functools import cache
from typing import Any


def streaming_scheduler_request(obj: Any) -> Any:
    """Return a shallow request copy that makes scheduler output observable."""
    scheduler_obj = copy.copy(obj)
    scheduler_obj.stream = True
    return scheduler_obj


def request_logs_metrics(obj: Any) -> bool:
    """Return whether this SGLang request should emit tokenizer metrics."""
    return not getattr(obj, "no_logs", False) and getattr(obj, "log_metrics", True)


def disable_request_metrics(obj: Any) -> None:
    """Suppress metrics across old ``no_logs`` and new ``log_metrics`` requests."""
    if hasattr(obj, "no_logs"):
        obj.no_logs = True
    if hasattr(obj, "log_metrics"):
        obj.log_metrics = False


def metric_suppression_kwargs(request_type: Any) -> dict[str, bool]:
    """Build compatible constructor kwargs for the installed SGLang request type."""
    try:
        parameters = inspect.signature(request_type).parameters
    except (TypeError, ValueError):
        return {}
    if "log_metrics" in parameters:
        return {"log_metrics": False}
    if "no_logs" in parameters:
        return {"no_logs": True}
    return {}


@cache
def _collector_method_accepts_keyword(
    collector_type: type[Any], method_name: str, keyword: str
) -> bool:
    """Return whether a collector method accepts a keyword argument.

    SGLang's tokenizer collector contract changes independently of the gRPC
    servicer. Inspect each collector class once so compatibility decisions do
    not add per-token reflection or mask ``TypeError`` raised inside a metric
    implementation.
    """
    try:
        parameters = inspect.signature(getattr(collector_type, method_name)).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _request_has_grammar(obj: Any) -> bool:
    sampling_params = getattr(obj, "sampling_params", None)
    grammar_fields = ("json_schema", "regex", "ebnf", "structural_tag")
    if isinstance(sampling_params, Mapping):
        return any(sampling_params.get(field) for field in grammar_fields)
    return any(getattr(sampling_params, field, None) for field in grammar_fields)


def observe_generation_metrics(
    collector: Any,
    state: Any,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    observe_ttft: bool,
) -> None:
    """Mirror SGLang tokenizer metrics for one gRPC scheduler output."""
    if collector is None or not request_logs_metrics(state.obj):
        return

    labels = dict(collector.labels)
    custom_labels = getattr(state.obj, "custom_labels", None)
    if isinstance(custom_labels, Mapping):
        labels.update({key: value for key, value in custom_labels.items() if key in labels})
    priority = getattr(state.obj, "priority", None)
    if "priority" in labels and priority is not None:
        labels["priority"] = str(priority)

    if not state.ttft_observed and observe_ttft:
        state.ttft_observed = True
        state.last_completion_tokens = completion_tokens
        ttft_kwargs = {}
        if _collector_method_accepts_keyword(
            type(collector), "observe_time_to_first_token", "stream"
        ):
            # ``state.obj`` retains the external client's requested mode. The
            # scheduler receives a forced-streaming copy solely so TTFT/TPOT
            # remain observable for non-streaming gRPC calls.
            ttft_kwargs["stream"] = bool(getattr(state.obj, "stream", False))
        collector.observe_time_to_first_token(
            labels,
            state.time_stats.get_first_token_latency(),
            **ttft_kwargs,
        )
    else:
        num_new_tokens = completion_tokens - state.last_completion_tokens
        if num_new_tokens > 0:
            collector.observe_inter_token_latency(
                labels,
                state.time_stats.get_interval(),
                num_new_tokens,
            )
            state.time_stats.set_last_time()
            state.last_completion_tokens = completion_tokens

    if state.finished:
        collector.observe_one_finished_request(
            labels,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            state.time_stats.get_e2e_latency(),
            _request_has_grammar(state.obj),
        )
