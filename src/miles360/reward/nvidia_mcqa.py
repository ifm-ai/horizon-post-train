import logging
import re

from .utils import timeout_limit

logger = logging.getLogger(__name__)

_FALLBACK_PATTERN = re.compile(r"(?i)Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*")


def _get_pattern(extra_info: dict):
    regex_str = extra_info.get("template_metadata")
    if regex_str:
        try:
            return re.compile(regex_str, re.IGNORECASE)
        except re.error:
            logger.warning("Invalid template_metadata regex %r, falling back", regex_str)
    return _FALLBACK_PATTERN


def _compute_score_impl(solution_str: str, ground_truth, extra_info: dict) -> dict:
    pattern = _get_pattern(extra_info)
    match = pattern.search(str(solution_str))
    if match is None:
        return {"score": 0.0, "acc": 0.0}
    answer = match.group(1).strip().upper()
    target = str(ground_truth).strip().upper()
    is_correct = answer == target
    return {"score": float(is_correct), "acc": float(is_correct)}


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, ground_truth, extra_info: dict) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info)


def compute_score(solution_str, ground_truth, extra_info, **kwargs) -> dict:
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
    except Exception:
        logger.exception("compute_score failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0}
