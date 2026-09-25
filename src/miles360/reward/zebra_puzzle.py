import asyncio
import ast
import json
import logging
import re

from miles.utils.types import Sample

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))

    if matches:
        final_answer = matches[-1].group(1).strip()
        try:
            return ast.literal_eval(final_answer)
        except (SyntaxError, ValueError):
            try:
                return json.loads(final_answer)
            except json.JSONDecodeError:
                return None
        except Exception:
            logger.exception("Failed to parse solution string %r", solution_str)
            return None
    return None


def compute_accuracy(answer, ground_truth):
    """Compare grid-level accuracy of the final answer with the ground truth."""
    if not isinstance(answer, dict):
        return 0

    num_rows = len(ground_truth["rows"])
    num_cols = len(ground_truth["header"])

    correct_cells = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                correct_cells += 1

    return correct_cells / (num_rows * num_cols)


def _compute_score_impl(solution_str, ground_truth):
    """Top-level (picklable) implementation of the zebra_puzzle scoring logic.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (common in CUDA / distributed training).
    """
    predicted_arrangement = extract_solution(solution_str)

    if predicted_arrangement is None:
        return 0.0
    try:
        return compute_accuracy(predicted_arrangement, ground_truth)
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return 0.0


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth):
    return _compute_score_impl(solution_str, ground_truth)


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute the reward score for zebra puzzle tasks.

    Args:
        solution_str (str): The model's response/solution string
        ground_truth (dict): The ground truth grid with 'header' and 'rows'
        extra_info: Unused, kept for interface compatibility

    Returns:
        dict: {"score": float, "acc": float}
    """
    try:
        score = _compute_score_with_timeout(solution_str, ground_truth)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        score = 0.0
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        score = 0.0

    return {"score": score, "acc": score}


# async def zebra_puzzle_rm(args, sample: Sample, **kwargs) -> float:
#     """Async reward function for miles.

#     Compatible with miles' --custom-rm-path interface.
#     Runs CPU-bound operations in a thread pool to avoid blocking the event loop.
#     Uses asyncio.wait_for for timeout to avoid unsafe fork() from within a thread
#     (multiprocessing-based timeout inside a thread pool risks deadlock).

#     Args:
#         args: Namespace with training arguments
#         sample: Sample object with prompt, response, label, metadata

#     Returns:
#         Reward score (float)
#     """
#     try:
#         result = await asyncio.wait_for(
#             asyncio.to_thread(
#                 compute_score,
#                 solution_str=sample.response,
#                 ground_truth=sample.label,
#                 extra_info=sample.metadata,
#             ),
#             timeout=10,
#         )
#     except asyncio.TimeoutError:
#         logger.warning("Computation timed out for sample with metadata=%r", sample.metadata)
#         return 0.0
#     except Exception:
#         logger.exception("compute_score failed for sample with metadata=%r", sample.metadata)
#         return 0.0

#     try:
#         return float(result["score"])
#     except (KeyError, TypeError, ValueError):
#         logger.warning("Invalid score returned: %r", result)
#         return 0.0


# async def batched_zebra_puzzle_rm(args, samples: list[Sample], max_concurrency: int = 8, **kwargs) -> list[float]:
#     """Batched async reward function for miles.

#     Compatible with miles' --custom-rm-path interface when using --group-rm.
#     Runs all samples concurrently using asyncio.gather with bounded concurrency.

#     Args:
#         args: Namespace with training arguments
#         samples: List of Sample objects

#     Returns:
#         List of reward scores
#     """
#     semaphore = asyncio.Semaphore(max_concurrency)

#     async def run_one(sample: Sample) -> float:
#         async with semaphore:
#             return await zebra_puzzle_rm(args, sample, **kwargs)

#     return await asyncio.gather(*(run_one(s) for s in samples))
