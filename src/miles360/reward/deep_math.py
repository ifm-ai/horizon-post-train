import ast
import logging
import re

from .utils import timeout_limit

logger = logging.getLogger(__name__)

ANSWER_PATTERNS = [
    r"(?:final answer|answer)[\s:]*(?:is)?[\s:]*([^\n.]+)",
    r"(?:evaluates to|equals to|is equal to)[\s:]*([^\n.]+)",
    r"therefore[\s,]+([^\n.]+)",
    r"thus[\s,]+([^\n.]+)",
    r"hence[\s,]+([^\n.]+)",
    r"=\s*([^\n]+)$",
    r"(?:limit|integral|sum|product)[\s\w]*(?:evaluates to|is|equals)[\s:]*([^\n.]+)",
]
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
BOXED_SPACE_PATTERN = re.compile(r"\\boxed\s+([^\s]+)")
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_AT_END_PATTERN = re.compile(r"(?:^|\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\d+/\d+)(?:\s*$|\s*[.,;]?\s*$)")
FRAC_PATTERN = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
ALLOWED_AST_NODES = (
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
    ast.Load,
    ast.Name,
)
EQUIVALENCES = [
    ("infinity", "\\infty"),
    ("inf", "\\infty"),
    ("undefined", "dne"),
    ("doesnotexist", "dne"),
    ("none", "dne"),
]
SAFE_NAMES = {
    "abs": abs,
    "min": min,
    "max": max,
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last answer from \\boxed{...} or \\boxed ... syntax."""
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1]

    matches = BOXED_SPACE_PATTERN.findall(text)
    if matches:
        return matches[-1]

    return None


def extract_answer_patterns(text: str) -> str | None:
    """Extract an answer from common textual answer patterns."""
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip().rstrip(".,;:")

    matches = NUMBER_AT_END_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()

    return None


def extract_solution(solution_str: str) -> str:
    """
    Extract a final answer from the model response.

    Prefers the last <answer>...</answer> block, then boxed answers, then
    common textual answer markers, and finally falls back to the raw text.
    """
    answer_matches = ANSWER_TAG_PATTERN.findall(solution_str)
    if answer_matches:
        return answer_matches[-1].strip()

    extracted_answer = extract_boxed_answer(solution_str)
    if extracted_answer is not None:
        return extracted_answer

    extracted_answer = extract_answer_patterns(solution_str)
    if extracted_answer is not None:
        return extracted_answer

    return solution_str.strip()


def _safe_eval_expression(expr: str) -> float | None:
    """Safely evaluate a simple numerical expression."""
    expr = expr.replace("\\pi", "pi").replace("\\e", "e").replace("^", "**")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in SAFE_NAMES:
            return None

    try:
        result = eval(compile(tree, "<deep_math>", "eval"), {"__builtins__": {}}, SAFE_NAMES)
    except Exception:
        return None

    try:
        return float(result)
    except Exception:
        return None


def normalize_fractions(text: str) -> str:
    """Normalize fraction representations into simple expression strings."""
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("\\dfrac", "\\frac")

    def frac_replacer(match):
        num, den = match.groups()
        num_val = _safe_eval_expression(num)
        den_val = _safe_eval_expression(den)
        if num_val is not None and den_val not in (None, 0.0):
            result = num_val / den_val
            if result == int(result):
                return str(int(result))
            return str(result)
        return f"({num})/({den})"

    return FRAC_PATTERN.sub(frac_replacer, text)


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical expressions for string and numeric comparison."""
    answer = str(answer).strip()
    answer = re.sub(r"\s+", "", answer)
    answer = answer.replace("$", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\Big", "")
    answer = answer.replace("\\big", "")
    answer = answer.replace("\\cdot", "*")
    answer = answer.replace("\\times", "*")
    answer = answer.replace("\\div", "/")
    answer = normalize_fractions(answer)
    return answer.rstrip(".,;:")


def is_equivalent(answer1: str, answer2: str) -> bool:
    """Check if two normalized answers are textually equivalent."""
    if answer1 == answer2:
        return True
    if answer1.lower() == answer2.lower():
        return True

    a1_lower = answer1.lower()
    a2_lower = answer2.lower()
    for eq1, eq2 in EQUIVALENCES:
        if (eq1 in a1_lower and eq2 in a2_lower) or (eq2 in a1_lower and eq1 in a2_lower):
            return True

    return False


def is_numerically_equivalent(answer1: str, answer2: str, tolerance: float = 1e-9) -> bool:
    """Check if two answers evaluate to the same numeric value."""
    val1 = _safe_eval_expression(answer1)
    val2 = _safe_eval_expression(answer2)
    if val1 is None or val2 is None:
        return False
    return abs(val1 - val2) < tolerance


def _compute_score_impl(solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Top-level (picklable) implementation of the DeepMath scorer."""
    del extra_info

    extracted_answer = extract_solution(str(solution_str))
    normalized_solution = normalize_math_answer(extracted_answer)
    normalized_ground_truth = normalize_math_answer(str(ground_truth))

    is_correct = is_equivalent(normalized_solution, normalized_ground_truth) or is_numerically_equivalent(
        normalized_solution,
        normalized_ground_truth,
    )
    score = float(is_correct)
    return {"score": score, "acc": score}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, ground_truth: str, extra_info=None) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info=extra_info)


def compute_score(solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """
    Compute the reward score for DeepMath solutions.

    Args:
        solution_str: The model's solution / answer.
        ground_truth: The correct answer from the dataset.
        extra_info: Unused, kept for interface compatibility.

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
