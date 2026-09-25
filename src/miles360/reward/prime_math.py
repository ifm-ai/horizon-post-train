"""
PRIME math reward function for miles.

Uses the comprehensive math_equal grader from prime_grader.py, combined with
miles' grade_answer_mathd and grade_answer_sympy for fast-path grading.

The grading flow matches the original prime_math compute_score:
1. match_answer() - extract answer using multiple patterns
2. grade_answer_mathd/sympy() - fast path using miles' graders
3. math_equal() - fallback for pi/complex cases

Usage:
    --custom-rm-path miles360.reward.prime_math:prime_math_rm

Or for batch mode with --group-rm:
    --custom-rm-path miles360.reward.prime_math:batched_prime_math_rm
"""

import asyncio
import logging
import math

from miles.rollout.rm_hub.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)
from miles.utils.types import Sample

from .prime_grader_lib.prime_grader_utils import match_answer, math_equal
from .utils import timeout_limit

logger = logging.getLogger(__name__)


def extract_answer_from_response(response: str) -> tuple[bool, str | None]:
    """Extract the answer from a model response using match_answer.

    Uses the sophisticated match_answer function which handles:
    - "answer:", "answer is", "answers are" markers
    - "is answer", "is the answer" markers
    - \\boxed{} extraction
    - Sentence ending cleanup
    - "be ", "is ", "are ", "=" markers

    For reasoning models with </think> tags, only looks at content after the tag.

    Args:
        response: The model's generated response

    Returns:
        Tuple of (is_matched, extracted_answer)
    """
    # For reasoning models, get content after </think>
    if "</think>" in response:
        response = response.split("</think>")[-1]

    # Use match_answer for sophisticated extraction
    is_matched, extracted = match_answer(response)

    # If match_answer didn't find anything useful, return None
    if not is_matched or not extracted:
        return False, None

    return is_matched, extracted


@timeout_limit(seconds=30.0)
def _compute_prime_math_reward_impl(
    response: str,
    label: str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
) -> dict:
    """Compute reward using PRIME math grader with miles utilities.

    Follows the original prime_math compute_score flow:
    1. match_answer() - extract answer using multiple patterns
    2. grade_answer_mathd/sympy() - fast path using miles' graders
    3. math_equal() - fallback for pi/complex cases

    Args:
        response: Model's generated response
        label: Ground truth answer
        include_percentage: Whether to check percentage variants (50 == 0.5*100)
        tolerance: Numerical tolerance for comparison

    Returns:
        Dict with:
            - score: 1.0 if correct, 0.0 if incorrect
            - acc: boolean correctness
            - extracted_answer: the answer extracted from response (for debugging)
    """
    # Step 0: Extract answer using match_answer (handles multiple patterns)
    is_matched, extracted_answer = extract_answer_from_response(response)

    if not extracted_answer:
        return {
            "score": 0.0,
            "acc": False,
            "extracted_answer": None,
        }

    # Process ground truth - extract from boxed if needed
    ground_truth = str(label)
    if "\\boxed" in ground_truth:
        ground_truth = extract_boxed_answer(ground_truth) or ground_truth

    # Step 1: Try miles' graders first (fast path)
    # This matches the original grade_answer() behavior
    if grade_answer_mathd(extracted_answer, ground_truth):
        return {
            "score": 1.0,
            "acc": True,
            "extracted_answer": extracted_answer,
        }

    if grade_answer_sympy(extracted_answer, ground_truth):
        return {
            "score": 1.0,
            "acc": True,
            "extracted_answer": extracted_answer,
        }

    # Step 2: Fall back to math_equal for pi/complex cases
    correct = False

    if "\\pi" in extracted_answer or "\\pi" in ground_truth:
        # Try multiple pi values
        for pi_val in [math.pi, 3.14]:
            if math_equal(
                extracted_answer,
                ground_truth,
                include_percentage=include_percentage,
                tolerance=tolerance,
                pi=pi_val,
            ):
                correct = True
                break
    else:
        correct = math_equal(
            extracted_answer,
            ground_truth,
            include_percentage=include_percentage,
            tolerance=tolerance,
        )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "extracted_answer": extracted_answer,
    }


def compute_prime_math_reward(
    response: str,
    label: str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
) -> dict:
    """Compute reward using PRIME math grader with miles utilities.

    Follows the original prime_math compute_score flow:
    1. match_answer() - extract answer using multiple patterns
    2. grade_answer_mathd/sympy() - fast path using miles' graders
    3. math_equal() - fallback for pi/complex cases

    Args:
        response: Model's generated response
        label: Ground truth answer
        include_percentage: Whether to check percentage variants (50 == 0.5*100)
        tolerance: Numerical tolerance for comparison

    Returns:
        Dict with:
            - score: 1.0 if correct, 0.0 if incorrect
            - acc: boolean correctness
            - extracted_answer: the answer extracted from response (for debugging)
    """
    try:
        return _compute_prime_math_reward_impl(
            response=response,
            label=label,
            include_percentage=include_percentage,
            tolerance=tolerance,
        )
    except TimeoutError:
        logger.warning("compute_prime_math_reward timed out for response=%r", response)
        return {"score": 0.0, "acc": False, "extracted_answer": None}
    except Exception:
        logger.exception("compute_prime_math_reward failed for response=%r", response)
        return {"score": 0.0, "acc": False, "extracted_answer": None}
