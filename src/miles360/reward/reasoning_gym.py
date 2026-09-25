import json
import logging
import re

import reasoning_gym as reasoning_gym_lib

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def _maybe_parse_json(value):
    """Parse a JSON string when possible; otherwise return the original value."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def extract_answer_from_solution(solution_str: str) -> str:
    """
    Extract the final answer from a model response.

    Prefers the last <answer>...</answer> block, then content after the last
    </think>, and finally falls back to the raw stripped response.
    """
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    think_pattern = r"</think>\s*(.*?)$"
    think_matches = re.findall(think_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if think_matches:
        answer = re.sub(r"<[^>]+>", "", think_matches[-1]).strip()
        if answer:
            return answer

    return solution_str.strip()


def is_valid_puzzle24_format(solution_str: str) -> bool:
    """Check whether a response resembles a mathematical puzzle24 expression."""
    solution = solution_str.strip()
    if not re.match(r"^[0-9+\-*/.() ]+$", solution):
        return False

    has_operator = any(op in solution for op in ["+", "-", "*", "/"])
    has_number = bool(re.search(r"\d", solution))
    return has_operator and has_number


def apply_task_specific_corrections(task: str, solution_str: str, ground_truth, raw_score: float) -> float:
    """Apply fixes for known issues in specific reasoning_gym tasks."""
    if task == "puzzle24":
        if raw_score == 0.01:
            return 0.01 if is_valid_puzzle24_format(solution_str) else 0.0
        return raw_score

    if task == "game_of_life_halting":
        return 1.0 if solution_str.strip().lower() == str(ground_truth).strip().lower() else 0.0

    return raw_score


def _build_task_context(ground_truth, extra_info=None, item=None) -> tuple[str, dict]:
    """Resolve the reasoning_gym task name and scorer entry payload."""
    task = None
    entry = None
    metadata = None
    extra_info_dict = _maybe_parse_json(extra_info)
    if not isinstance(extra_info_dict, dict):
        extra_info_dict = {}

    task = extra_info_dict.get("task")
    entry = extra_info_dict.get("entry")

    metadata = _maybe_parse_json(extra_info_dict.get("metadata"))
    if metadata is not None and not isinstance(metadata, dict):
        metadata = {}

    if not task and isinstance(item, dict):
        task = item.get("ability")

    if not task and isinstance(ground_truth, dict):
        task = ground_truth.get("task")
        entry = ground_truth

    if not task:
        raise ValueError("task must be provided in extra_info, item, or ground_truth dict.")

    if entry is None:
        entry = {"answer": ground_truth}
    elif isinstance(entry, dict):
        entry = dict(entry)
    else:
        entry = {"answer": entry}

    entry_metadata = entry.get("metadata")
    if not isinstance(entry_metadata, dict):
        entry_metadata = {}
    else:
        entry_metadata = dict(entry_metadata)

    if metadata:
        entry_metadata.update(metadata)

    entry["metadata"] = entry_metadata
    return task, entry


def _compute_score_impl(solution_str, ground_truth, extra_info=None, item=None) -> dict:
    """Top-level (picklable) implementation of the reasoning_gym scorer."""
    task, entry = _build_task_context(ground_truth=ground_truth, extra_info=extra_info, item=item)
    scorer = reasoning_gym_lib.get_score_answer_fn(task)

    entry["metadata"]["task"] = task
    entry["metadata"]["solution_str"] = solution_str
    entry["metadata"]["ground_truth"] = ground_truth
    if extra_info is not None:
        entry["metadata"]["extra_info"] = extra_info
    if item is not None:
        entry["metadata"]["item"] = item

    clean_answer = extract_answer_from_solution(str(solution_str))
    raw_score = scorer(answer=clean_answer, entry=entry)
    corrected_score = apply_task_specific_corrections(task, clean_answer, ground_truth, raw_score)

    logger.debug(
        "reasoning_gym score task=%s raw_score=%s corrected_score=%s clean_answer=%r",
        task,
        raw_score,
        corrected_score,
        clean_answer,
    )
    score = float(corrected_score)
    return {"score": score, "acc": score}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth, extra_info=None, item=None) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info=extra_info, item=item)


def compute_score(solution_str, ground_truth, extra_info=None, item=None) -> dict:
    """
    Compute the reward score for reasoning_gym tasks.

    Args:
        solution_str: The model's response string.
        ground_truth: The expected answer or full entry dict.
        extra_info: Optional metadata containing task / entry information.
        item: Optional fallback dataset item.

    Returns:
        dict: {"score": float, "acc": float}
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info=extra_info, item=item)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
