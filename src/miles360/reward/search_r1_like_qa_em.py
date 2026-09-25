# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import logging
import re
import string

from .utils import timeout_limit

logger = logging.getLogger(__name__)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer span from the last <answer>...</answer> block."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def _compute_score_impl(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Top-level (picklable) implementation of the Search-R1 EM scorer."""
    del method

    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    if answer is None:
        return 0.0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def _compute_score_subem_impl(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Top-level (picklable) implementation of the Search-R1 substring EM scorer."""
    del method

    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    return _compute_score_impl(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )


@timeout_limit(seconds=30.0)
def _compute_score_subem_with_timeout(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    return _compute_score_subem_impl(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, extra_info=None):
    """The scoring function for exact match (EM)."""
    del extra_info

    try:
        return _compute_score_with_timeout(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method=method,
            format_score=format_score,
            score=score,
        )
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return 0.0
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return 0.0


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, extra_info=None):
    """The scoring function for substring exact match (EM)."""
    del extra_info

    try:
        return _compute_score_subem_with_timeout(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method=method,
            format_score=format_score,
            score=score,
        )
    except TimeoutError:
        logger.warning("compute_score_subem timed out for solution_str=%r", solution_str)
        return 0.0
    except Exception:
        logger.exception("_compute_score_subem_impl failed for solution_str=%r", solution_str)
        return 0.0
