"""
Tabular reasoning reward function for miles.
"""

import ast
import logging
import re

from .prime_grader_lib.prime_grader_utils import math_equal
from .utils import is_equiv, remove_boxed, timeout_limit

logger = logging.getLogger(__name__)


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def drop_latex_text(answer: str) -> str:
    """Replace simple LaTeX text fragments with plain text."""
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = answer.replace("\\", "")
    return answer.strip()


def normalize_answer_text(answer: str) -> str:
    """Normalize a raw answer string before comparison."""
    return drop_latex_text(str(answer).strip().lower())


def _safe_numeric_eval(answer: str) -> float | None:
    """Evaluate a limited arithmetic expression used in table answers."""
    normalized = (
        answer.replace(",", "")
        .replace("%", " / 100")
        .replace("$", "")
        .replace(":", "/")
        .replace("\\", "")
        .strip()
    )

    try:
        expr = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    if not all(isinstance(node, allowed_nodes) for node in ast.walk(expr)):
        return None

    try:
        return float(eval(compile(expr, "<tabular_reasoning>", "eval"), {"__builtins__": None}, {}))
    except Exception:
        return None


def _check_single_answer(answer: str, ground_truth: str) -> bool:
    """Compare a single answer token against the ground truth."""
    numeric_answer = _safe_numeric_eval(answer)
    if numeric_answer is not None:
        try:
            return bool(math_equal(numeric_answer, ground_truth, tolerance=1e-3))
        except Exception:
            logger.exception("math_equal failed for answer=%r ground_truth=%r", answer, ground_truth)

    return bool(is_equiv(answer, ground_truth))


def extract_solution(model_output: str) -> str:
    """Extract the final answer span from the model output."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, model_output, flags=re.DOTALL))
    if matches:
        answer = matches[-1].group(1).strip()
    else:
        answer = model_output.split("</think>")[-1].strip()

    boxed_answer = last_boxed_only_string(answer)
    if boxed_answer is not None:
        answer = remove_boxed(boxed_answer)

    return normalize_answer_text(answer)


def _compare_multi_answer(prediction: str, ground_truth: str) -> bool:
    """Compare pipe-separated multi-part answers order-insensitively."""
    expected_parts = sorted(normalize_answer_text(ans) for ans in ground_truth.split("|"))
    predicted_parts = sorted(normalize_answer_text(ans) for ans in prediction.split("|"))

    if len(expected_parts) != len(predicted_parts):
        return False

    return all(_check_single_answer(predicted, expected) for predicted, expected in zip(predicted_parts, expected_parts))


def _compute_score_impl(model_output: str, ground_truth: str) -> dict:
    """Top-level (picklable) implementation of the tabular reasoning scorer."""
    prediction = extract_solution(str(model_output))
    target = normalize_answer_text(str(ground_truth))

    if "|" in target:
        score = _compare_multi_answer(prediction, target)
    else:
        score = _check_single_answer(prediction, target)

    value = float(bool(score))
    return {"score": value, "acc": value}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(model_output: str, ground_truth: str) -> dict:
    return _compute_score_impl(model_output, ground_truth)


def compute_score(model_output: str, ground_truth: str, extra_info=None) -> dict:
    """
    Compute the reward score for tabular reasoning tasks.

    Args:
        model_output: The model's response string.
        ground_truth: The expected answer string.
        extra_info: Unused, kept for interface compatibility.

    Returns:
        dict: {"score": float, "acc": float}
    """
    del extra_info

    try:
        return _compute_score_with_timeout(model_output, ground_truth)
    except TimeoutError:
        logger.warning("compute_score timed out for model_output=%r", model_output)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for model_output=%r", model_output)
        return {"score": 0.0, "acc": 0.0}
