import re
import ast
import logging
import numpy as np

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, flags=re.DOTALL))
    if matches:
        final_answer = matches[-1].group(1).strip()
        final_answer = final_answer.replace("\n", "")
        final_answer = final_answer.replace("...", "-1")
        try:
            start = final_answer.index("[[")
            end = final_answer.index("]]", start) + 2
            array_str = final_answer[start:end]
            array = ast.literal_eval(array_str)
            if all(isinstance(i, list) for i in array):
                return array
            else:
                return [[0]]
        except Exception:
            return [[0]]
    return [[0]]


def pad_array_with_value(array, target_shape, pad_value):
    """
    Pad the given array to the target shape with the specified pad value.

    Places the smaller array at the top-left corner of the target shape,
    padding at the ends so that correct-pixel counts are unambiguous.

    Parameters:
        array (list): The original array to be padded.
        target_shape (tuple): The desired shape (rows, columns).
        pad_value (int): The value to use for padding.

    Returns:
        np.ndarray: Padded array with the specified target shape.
    """
    padded_array = np.full(target_shape, pad_value, dtype=int)
    try:
        array = np.stack(array).astype(int)
    except Exception:
        array = np.array([[0]])
    original_shape = array.shape
    padded_array[: original_shape[0], : original_shape[1]] = array
    return padded_array


def compare_solutions_with_padding(generated_output, correct_output, pad_value=-1):
    """
    Compare the generated output with the correct output using padding to align shapes.

    Parameters:
        generated_output (list): The generated solution array.
        correct_output (list): The correct solution array.
        pad_value (int): The value used for padding (default -1, which must not
            appear in any valid solution).

    Returns:
        tuple[float, float]: (is_correct, correct_percentage)
            is_correct is 1.0 if the solutions match exactly, 0.0 otherwise.
            correct_percentage is the fraction of correctly matched non-pad pixels.
    """
    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(len(generated_output[0]), len(correct_output[0]))
    target_shape = (max_rows, max_cols)

    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)

    total_pixels = max_rows * max_cols
    correct_pixels = np.sum(
        (padded_generated == padded_correct)
        & (padded_generated != pad_value)
        & (padded_correct != pad_value)
    )
    correct_percentage = correct_pixels / total_pixels
    is_correct = float(correct_pixels == total_pixels)
    return is_correct, correct_percentage


def _compute_score_impl(model_output: str, ground_truth) -> dict:
    """Top-level (picklable) implementation of the arcagi scoring logic.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (common in CUDA / distributed training).
    """
    final_answer = extract_solution(str(model_output))
    is_correct, correct_percentage = compare_solutions_with_padding(final_answer, ground_truth)
    return {"score": is_correct, "acc": is_correct}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(model_output: str, ground_truth) -> dict:
    return _compute_score_impl(model_output, ground_truth)


def compute_score(model_output: str, ground_truth: np.ndarray, extra_info=None) -> dict:
    """
    Compute the reward score for ARC-AGI tasks.

    Args:
        model_output (str): The model's response string
        ground_truth (np.ndarray or list): The correct output grid
        extra_info: Unused, kept for interface compatibility

    Returns:
        dict: {"score": float, "acc": float}
    """
    try:
        return _compute_score_with_timeout(model_output, ground_truth)
    except TimeoutError:
        logger.warning("compute_score timed out for model_output=%r", model_output)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for model_output=%r", model_output)
        return {"score": 0.0, "acc": 0.0}
