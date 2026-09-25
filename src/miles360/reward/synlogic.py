"""
SynLogic reward function for miles.

Dispatches to the appropriate puzzle verifier based on the data_source name,
which is passed via kwargs (e.g. "synlogic_arrow_maze" → ArrowMazeVerifier).

All 25 active verifiers are supported:
    arrow_maze, boolean_expressions, buggy_tables, campsite,
    dyck_language, dyck_language_errors, dyck_language_reasoning_errors,
    goods_exchange, math_path, minesweeper, norinori, number_wall, numbrix,
    object_counting, object_properties, operation, skyscraper_puzzle,
    space_reasoning, space_reasoning_tree, star_placement_puzzle,
    time_sequence, web_of_lies, word_sorting, word_sorting_mistake,
    wordscapes

Routing is handled by the __init__.py dispatcher — this module assumes all
samples it receives are already synlogic samples. The task type is read from
sample.metadata["data_source"] per sample, so mixed batches are supported.

Usage (via __init__.py router):
    --custom-rm-path miles360.reward:synlogic_rm

Or direct (single task only):
    --custom-rm-path miles360.reward.synlogic:synlogic_rm
"""

import asyncio
import logging

from miles.utils.types import Sample

from .utils import timeout_limit

logger = logging.getLogger(__name__)

from .synlogic_lib.data import Data
from .synlogic_lib.synlogic import verifier_classes


def _compute_score_impl(solution_str: str, extra_info: dict, data_source: str) -> dict:
    """Core scoring logic for SynLogic tasks.

    Module-level (picklable) so it is safe to pass to asyncio.to_thread /
    multiprocessing without closure-pickling issues.

    Replicates the original dispatch logic:
        form_solution = solution_str after </think> tag
        data = Data.from_json_str(extra_info["game_data_str"])
        verifier = verifier_classes[task_name]()
        score = 1.0 if verifier.verify(data, form_solution) else 0.0

    Args:
        solution_str: The model's raw response string.
        extra_info: Dict containing "game_data_str" (JSON-serialised Data object).
        data_source: The full data source name, e.g. "synlogic_arrow_maze".
            The "synlogic_" prefix is stripped to obtain the task key.

    Returns:
        Dict with keys "score" and "acc" (both float).

    Raises:
        KeyError: If data_source maps to an unknown verifier.
        KeyError: If extra_info does not contain "game_data_str".
    """
    task_name = data_source.replace("synlogic_", "")
    if task_name not in verifier_classes:
        raise KeyError(
            f"Unknown synlogic task '{task_name}'. "
            f"Available: {sorted(verifier_classes)}"
        )

    # Strip chain-of-thought — only score the part after </think>
    form_solution = solution_str.strip().split("</think>")[-1].strip()

    data = Data.from_json_str(extra_info["game_data_str"])
    verifier = verifier_classes[task_name]()
    res = verifier.verify(data, form_solution)
    score = 1.0 if res else 0.0
    return {"score": score, "acc": score}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, extra_info: dict, data_source: str) -> dict:
    return _compute_score_impl(solution_str, extra_info, data_source)


def compute_score(solution_str: str, ground_truth=None, extra_info: dict = None, data_source: str = "") -> dict:
    """Synchronous entry point — wraps _compute_score_impl with error handling.

    Args:
        solution_str: The model's raw response string.
        ground_truth: Unused; the ground truth is embedded in extra_info["game_data_str"].
        extra_info: Dict containing "game_data_str".
        data_source: The full data source name, e.g. "synlogic_arrow_maze".

    Returns:
        Dict with keys "score" and "acc".
    """
    try:
        return _compute_score_with_timeout(solution_str, extra_info or {}, data_source)
    except TimeoutError:
        logger.warning("compute_score timed out for data_source=%r", data_source)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("_compute_score_impl failed for data_source=%r", data_source)
        return {"score": 0.0, "acc": 0.0}
