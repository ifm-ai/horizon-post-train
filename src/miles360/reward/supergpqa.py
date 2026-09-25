import logging
import re

from .utils import timeout_limit

logger = logging.getLogger(__name__)

OPTION_LETTERS = tuple("ABCDEFGHIJ")
OPTION_CLASS = "A-J"
STOP_WORDS = ("</s>", "<|im_end|>", "<|endoftext|>")
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
FINAL_ANSWER_PATTERNS = (
    re.compile(r"(?i)final answer\s*:\s*(.+?)(?:\n|$)"),
    re.compile(r"(?i)the answer is\s*:\s*(.+?)(?:\n|$)"),
    re.compile(r"(?i)answer\s*:\s*(.+?)(?:\n|$)"),
)
STRICT_OPTION_PATTERNS = (
    re.compile(rf"^\(?\s*([{OPTION_CLASS}])\s*\)?\.?$"),
    re.compile(rf"(?<![A-Z])\(([{OPTION_CLASS}])\)(?![A-Z])"),
    re.compile(rf"(?<![A-Z])([{OPTION_CLASS}])(?![A-Z])"),
)


def _strip_chat_wrapper(solution_str: str) -> str:
    """Remove common chat-template wrappers from model output."""
    if "<|im_start|>user" in solution_str:
        return re.sub(
            r"^.*?<\|im_start\|>assistant",
            "<|im_start|>assistant",
            solution_str,
            flags=re.DOTALL,
            count=1,
        )
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:")[-1].strip()
    return solution_str


def _truncate_at_stop_words(text: str) -> str:
    """Trim generation stop tokens from a response."""
    for stop_word in STOP_WORDS:
        if stop_word in text:
            text = text.split(stop_word)[0].strip()
    return text


def extract_last_boxed(text: str) -> str | None:
    """Extract the last boxed expression from the text."""
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_last_final_answer(text: str) -> str | None:
    """Extract the last final-answer style span from the text."""
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
    return None


def extract_solution(solution_str: str) -> str | None:
    """Extract the answer-bearing span from a SuperGPQA model response."""
    model_output = _truncate_at_stop_words(_strip_chat_wrapper(str(solution_str)))

    tag_matches = list(ANSWER_TAG_PATTERN.finditer(model_output))
    if tag_matches:
        return tag_matches[-1].group(1).strip()

    boxed_answer = extract_last_boxed(model_output)
    if boxed_answer:
        return boxed_answer

    final_answer = extract_last_final_answer(model_output)
    if final_answer:
        return final_answer

    return model_output.strip() or None


def get_prediction(output: str) -> str | None:
    """Extract a deterministic multiple-choice prediction from the output."""
    solution = extract_solution(output)
    if solution is None:
        return None

    solution = solution.upper().strip()
    for pattern in STRICT_OPTION_PATTERNS:
        matches = pattern.findall(solution)
        if matches:
            return matches[-1]

    return None


def _compute_score_impl(
    solution_str: str,
    ground_truth,
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    """Top-level (picklable) implementation of the SuperGPQA scorer."""
    answer = get_prediction(solution_str)
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
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    return _compute_score_impl(
        solution_str=solution_str,
        ground_truth=ground_truth,
        format_score=format_score,
        score=score,
    )


def compute_score(
    solution_str,
    ground_truth,
    extra_info: any = None,
    method="strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    """
    Compute the reward score for SuperGPQA multiple-choice tasks.

    Args:
        solution_str: The model's response text.
        ground_truth: The correct answer choice.
        extra_info: Unused, kept for interface compatibility.
        method: Unused compatibility argument retained for API parity.
        format_score: Score to assign when an answer is extracted but incorrect.
        score: Score to assign when the extracted answer is correct.

    Returns:
        dict: {"score": float, "acc": float}
    """
    del extra_info, method

    try:
        return _compute_score_with_timeout(
            solution_str=solution_str,
            ground_truth=ground_truth,
            format_score=format_score,
            score=score,
        )
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
