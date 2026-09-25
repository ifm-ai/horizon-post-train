import json
import logging

from .livebench_lib.data_analysis.cta.utils import cta_process_results
from .livebench_lib.data_analysis.tablejoin.utils import joinmap_process_results
from .livebench_lib.data_analysis.tablereformat.utils import table_process_results
from .livebench_lib.reasoning.house_traversal.utils import house_traversal_process_results
from .livebench_lib.reasoning.spatial.utils import spatial_process_results
from .livebench_lib.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from .livebench_lib.reasoning.web_of_lies_v3.utils import web_of_lies_v3_process_results
from .livebench_lib.reasoning.zebra_puzzle.utils import zebra_puzzle_process_results_old
from .livebench_lib.writing.connections.utils import connections_process_results_old
from .livebench_lib.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from .livebench_lib.writing.typos.utils import typos_process_results
from .utils import timeout_limit

logger = logging.getLogger(__name__)


def _maybe_parse_json(value):
    """Parse JSON strings when possible; otherwise return the original value."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_extra_info(extra_info) -> dict:
    """Normalize extra_info into a dictionary."""
    extra_info = _maybe_parse_json(extra_info)
    if isinstance(extra_info, dict):
        return extra_info
    return {}


def _extract_task(extra_info: dict, ground_truth) -> str | None:
    """Resolve the LiveBench task identifier."""
    task = extra_info.get("task")
    if task:
        return task
    if isinstance(ground_truth, dict):
        return ground_truth.get("task")
    return None


def _extract_ground_truth_value(ground_truth):
    """Unwrap dict-style ground truth payloads when needed."""
    if isinstance(ground_truth, dict):
        for key in ("answer", "ground_truth", "target", "label"):
            if key in ground_truth:
                return ground_truth[key]
    return ground_truth


def _extract_tableformat_prompt(extra_info: dict, ground_truth) -> str | None:
    """Resolve the input command needed for the tableformat scorer."""
    for key in ("input_command", "prompt", "question", "instruction", "query"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value

    if isinstance(ground_truth, dict):
        for key in ("input_command", "prompt", "question", "instruction", "query"):
            value = ground_truth.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return None


def _compute_score_impl(solution_str, ground_truth, extra_info=None) -> dict:
    """Top-level (picklable) implementation of the LiveBench scoring logic."""
    extra_info_dict = _normalize_extra_info(extra_info)
    task = _extract_task(extra_info_dict, ground_truth)
    if not task:
        logger.warning("Missing LiveBench task in extra_info=%r ground_truth=%r", extra_info, ground_truth)
        return {"score": 0.0, "acc": 0.0}

    target = _extract_ground_truth_value(ground_truth)

    if task == "cta":
        score = cta_process_results(target, solution_str)
    elif task == "tablejoin":
        score = joinmap_process_results(target, solution_str)
    elif task == "tableformat":
        input_command = _extract_tableformat_prompt(extra_info_dict, ground_truth)
        if not input_command:
            logger.warning("Missing tableformat input command for extra_info=%r ground_truth=%r", extra_info, ground_truth)
            return {"score": 0.0, "acc": 0.0}
        version = extra_info_dict.get("version", "v1")
        score = table_process_results(input_command, target, solution_str, version=version)
    elif task == "web_of_lies_v2":
        score = web_of_lies_process_results(target, solution_str)
    elif task == "web_of_lies_v3":
        score = web_of_lies_v3_process_results(target, solution_str)
    elif task == "house_traversal":
        score = house_traversal_process_results(target, solution_str)
    elif task == "zebra_puzzle":
        score = zebra_puzzle_process_results_old(target, solution_str)
    elif task == "spatial":
        score = spatial_process_results(target, solution_str)
    elif task == "plot_unscrambling":
        score = plot_unscrambling_process_results(target, solution_str)
    elif task == "typos":
        score = typos_process_results(target, solution_str)
    elif task == "connections":
        score = connections_process_results_old(target, solution_str)
    else:
        logger.warning("Unsupported LiveBench task=%r", task)
        score = 0.0

    score = float(score)
    return {"score": score, "acc": score}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth, extra_info=None) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info=extra_info)


def compute_score(solution_str, ground_truth, extra_info=None) -> dict:
    """
    Compute the reward score for LiveBench tasks.

    Args:
        solution_str: The model's response string.
        ground_truth: The task answer or task-specific payload.
        extra_info: Metadata containing at least the LiveBench task name.

    Returns:
        dict: {"score": float, "acc": float}
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info=extra_info)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
