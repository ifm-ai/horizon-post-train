import logging
import re

from .utils import timeout_limit

logger = logging.getLogger(__name__)

STRICT_ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
FLEXIBLE_ANSWER_PATTERN = re.compile(r"\(([A-D])\)")
VALID_METHODS = {"strict", "flexible"}


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """
    Extract the final GPQA answer choice from a model response.

    Args:
        solution_str: The full text response from the model.
        method: Either "strict" or "flexible".

    Returns:
        The extracted answer choice ("A" through "D"), or None if not found.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported extraction method: {method}")

    solution_str = str(solution_str)

    if method == "strict":
        match = STRICT_ANSWER_PATTERN.search(solution_str)
        return match.group(1).upper() if match else None

    answers = FLEXIBLE_ANSWER_PATTERN.findall(solution_str.upper())
    return answers[-1] if answers else None


def _compute_score_impl(
    solution_str: str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    """Top-level (picklable) implementation of the GPQA scoring logic."""
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return {"score": 0.0, "acc": 0.0}

    target = str(ground_truth).strip().upper()
    is_correct = answer == target
    return {
        "score": float(score if is_correct else format_score),
        "acc": float(is_correct),
    }


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(
    solution_str: str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    return _compute_score_impl(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )


def compute_score(
    solution_str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    extra_info=None,
):
    """
    Compute the reward score for GPQA multiple-choice tasks.

    Reference:
        Trung, Luong, et al. "ReFT: Reasoning with Reinforced Fine-Tuning."

    Args:
        solution_str: The model's response text.
        ground_truth: The correct answer choice.
        method: Extraction mode, either "strict" or "flexible".
        format_score: Score to assign when an answer is extracted but incorrect.
        score: Score to assign when the extracted answer is correct.
        extra_info: Unused, kept for interface compatibility.

    Returns:
        dict: {"score": float, "acc": float}
    """
    del extra_info

    try:
        return _compute_score_with_timeout(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method=method,
            format_score=format_score,
            score=score,
        )
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
