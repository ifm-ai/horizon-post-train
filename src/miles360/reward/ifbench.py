import ast
import json
import asyncio
import logging

import numpy as np

from .ifeval_lib.instructions_util import download_nltk_resources
from .utils import timeout_limit

logger = logging.getLogger(__name__)

download_nltk_resources()

from .ifbench_lib.instructions_registry import INSTRUCTION_DICT


def _compute_score_impl(solution_str, ground_truth):
    """Top-level (picklable) implementation of the IFBench scoring logic.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (common in CUDA / distributed training).
    """
    # Strip off any thinking section
    if "</think>" in solution_str:
        answer = solution_str.split("</think>", 1)[1].strip()
    else:
        answer = solution_str.strip()

    # Parse ground_truth if it's a string
    if isinstance(ground_truth, str):
        try:
            gt_list = ast.literal_eval(ground_truth)
        except Exception:
            gt_list = json.loads(ground_truth)
    else:
        gt_list = ground_truth

    # Take the first set of constraints
    if not isinstance(gt_list, list) or not gt_list:
        return {"score": 0.0, "acc": False}
    first_item = gt_list[0]
    instruction_ids = first_item.get("instruction_id", [])
    kwargs_list = first_item.get("kwargs", [])

    # Guard: empty constraint list must not silently score 1.0
    # (all([]) is True in Python, which would be incorrect)
    if not instruction_ids:
        return {"score": 0.0, "acc": False}

    # Evaluate each instruction
    results = []
    for instr_id, raw_args in zip(instruction_ids, kwargs_list):
        # Prepare args dict
        args = {} if raw_args is None else raw_args
        # Convert numpy and floats
        clean_args = {}
        for key, val in args.items():
            if isinstance(val, float):
                clean_args[key] = int(val)
            elif isinstance(val, np.ndarray):
                clean_args[key] = val.tolist()
            else:
                clean_args[key] = val

        # Build and check instruction
        instr_cls = INSTRUCTION_DICT[instr_id]
        instr = instr_cls(instr_id)
        instr.build_description(**clean_args)
        passed = bool(answer and instr.check_following(answer))
        results.append(passed)

    score = 1.0 if all(results) else 0.0
    return {"score": score, "acc": score == 1.0}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth):
    return _compute_score_impl(solution_str, ground_truth)


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute the reward score for IFBench tasks based on ground truth constraints.

    Args:
        solution_str (str): Model's full output, may include a '<think>' section.
        ground_truth (str or list): Original ground_truth, either a Python-literal string or list of dicts.
        extra_info (dict, optional): Ignored for IFBench since constraints are in ground_truth.

    Returns:
        dict: {"score": float, "acc": bool}
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
