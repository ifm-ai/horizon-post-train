"""
Logic graph reward function for miles.

Scores model responses on graph reasoning tasks where the answer is a single
alphabetic token (e.g. a node name or yes/no label) that must exactly match
the ground truth.

Usage:
    --custom-rm-path miles360.reward.logic_graph:logic_graph_rm

Or for batch mode with --group-rm:
    --custom-rm-path miles360.reward.logic_graph:batched_logic_graph_rm
"""

import asyncio
import logging
import re

from miles.utils.types import Sample

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def extract_solution(solution_str: str) -> str | None:
    """Extract the answer token from the model's response.

    Looks for content inside the last <answer>…</answer> tag and returns it
    only if it consists entirely of ASCII letters (e.g. a node name or a
    yes/no label).

    Args:
        solution_str: Raw model response string.

    Returns:
        The extracted answer string, or None if not found / invalid format.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None

    final_answer = matches[-1].group(1).strip()
    if re.search(r"^[A-Za-z]+$", final_answer):
        return final_answer

    return None


def _compute_score_impl(solution_str: str, ground_truth) -> dict:
    """Core scoring logic for logic-graph tasks.

    Module-level (picklable) so it is safe to pass to asyncio.to_thread /
    multiprocessing without closure-pickling issues.

    The answer is correct if and only if the extracted token is an exact
    case-insensitive match of the ground truth.

    Args:
        solution_str: The model's raw response string.
        ground_truth: The expected answer (str or coercible to str).

    Returns:
        Dict with keys "score" and "acc" (both float).
    """
    target = ground_truth.lower() if isinstance(ground_truth, str) else str(ground_truth).lower()
    solution = extract_solution(solution_str)

    if solution is None:
        return {"score": 0.0, "acc": 0.0}

    score = 1.0 if solution.lower() == target else 0.0
    return {"score": score, "acc": score}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, ground_truth) -> dict:
    return _compute_score_impl(solution_str, ground_truth)


def compute_score(solution_str: str, ground_truth, extra_info=None) -> dict:
    """Synchronous entry point — wraps _compute_score_impl with error handling.

    Args:
        solution_str: The model's raw response string.
        ground_truth: The expected answer.
        extra_info: Unused; kept for interface compatibility.

    Returns:
        Dict with keys "score" and "acc".
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
