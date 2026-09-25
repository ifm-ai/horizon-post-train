"""
Logic puzzle reward function for miles.

Scores ordering-style logic puzzle outputs where the model is expected to
return a final list inside the last <answer>...</answer> block.
"""

import ast
import logging
import re

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def _normalize_sequence(value):
    """Normalize a predicted or ground-truth sequence for comparison."""
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        return None

    return [str(item).strip().lower() for item in value]


def extract_solution(solution_str: str) -> list[str] | None:
    """Extract the final ordered list from the last <answer> block."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, flags=re.DOTALL))
    if not matches:
        return None

    final_answer = matches[-1].group(1).strip()
    list_matches = list(re.finditer(r"\[[^\[\]]*\]", final_answer, flags=re.DOTALL))
    candidate = list_matches[-1].group(0).strip() if list_matches else final_answer

    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        parsed = None
    except Exception:
        logger.exception("Failed to parse candidate answer %r", candidate)
        return None

    normalized = _normalize_sequence(parsed)
    if normalized is not None:
        return normalized

    if candidate.startswith("[") and candidate.endswith("]"):
        inner = candidate[1:-1].strip()
        if not inner:
            return []

        tokens = []
        for part in inner.split(","):
            token = part.strip().strip("'\"").strip()
            if token:
                tokens.append(token.lower())

        return tokens or None

    return None


def compute_edit_distance(list1: list[str], list2: list[str]) -> int:
    """Calculate the Levenshtein distance between two lists."""
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]

    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j

    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    return dp[len(list1)][len(list2)]


def _compute_score_impl(solution_str: str, ground_truth, method: str = "strict") -> dict:
    """Top-level (picklable) implementation of the logic-puzzle scoring logic."""
    target = _normalize_sequence(ground_truth)
    predicted_arrangement = extract_solution(solution_str)

    if predicted_arrangement is None or target is None:
        return {"score": 0.0, "acc": 0.0}

    if predicted_arrangement == target:
        return {"score": 1.0, "acc": 1.0}

    if method != "strict":
        edit_distance = compute_edit_distance(predicted_arrangement, target)
        max_possible_dist = max(len(predicted_arrangement), len(target), 1)
        score = max(0.0, 1.0 - (edit_distance / max_possible_dist))
        return {"score": score, "acc": score}

    return {"score": 0.0, "acc": 0.0}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, ground_truth, method: str = "strict") -> dict:
    return _compute_score_impl(solution_str, ground_truth, method=method)


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info=None,
    method: str = "strict",
    timeout: float = 10.0,
) -> dict:
    """
    Compute the reward score for ordering logic puzzle tasks.

    Args:
        solution_str: The model's response string.
        ground_truth: The expected ordered list.
        extra_info: Unused, kept for interface compatibility.
        method: "strict" for exact-match scoring, anything else for edit-distance
            based partial credit.
        timeout: Unused compatibility argument retained from the original module.

    Returns:
        dict: {"score": float, "acc": float}
    """
    del extra_info, timeout

    try:
        return _compute_score_with_timeout(solution_str, ground_truth, method=method)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
