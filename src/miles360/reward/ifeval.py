import asyncio
import logging

import numpy as np

from .utils import timeout_limit

logger = logging.getLogger(__name__)

from .ifeval_lib import instructions_registry


def _compute_score_impl(solution_str, ground_truth, extra_info):
    """Top-level (picklable) implementation of the IFEval scoring logic.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (common in CUDA / distributed training).
    """
    if "</think>" in solution_str:
        answer = solution_str.split("</think>")[1]
    else:
        answer = solution_str

    instruction_id_list = extra_info["instruction_id_list"]

    # Guard: empty constraint list must not silently score 1.0
    # (all([]) is True in Python, which would be incorrect)
    if not instruction_id_list:
        return {"score": 0.0, "acc": False}

    is_following_list = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTIfON_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors
        # in build_description. Convert numpy arrays to lists and floats to ints.
        kwargs = {
            k: int(v) if isinstance(v, float) else v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in ground_truth[index].items()
            if v is not None
        }

        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=extra_info["prompt"])

        is_following_list.append(bool(answer.strip() and instruction.check_following(answer)))

    acc = all(is_following_list)
    return {"score": 1.0 if acc else 0.0, "acc": acc}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth, extra_info):
    return _compute_score_impl(solution_str, ground_truth, extra_info)


def compute_score(solution_str, ground_truth, extra_info):
    """
    Compute the reward score for IFEval tasks.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
    ACL 2024.

    Args:
        solution_str (str): Model's full output, may include a '<think>' section.
        ground_truth (list): List of per-instruction kwargs dicts.
        extra_info (dict): Must contain 'instruction_id_list' and optionally 'prompt'.

    Returns:
        dict: {"score": float, "acc": bool}
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
