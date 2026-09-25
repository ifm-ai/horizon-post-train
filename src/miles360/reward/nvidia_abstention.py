"""
Reward scorer for nvidia/Nemotron-RL-QA-Abstention-v1.

The model is prompted to answer with \\boxed{answer} or \\boxed{[IDK]} when
uncertain. Correctness of the answers is determined by an LLM judge.

Reward shaping:
    correct answer  → +1.0
    [IDK] abstain   →  0.0  (neutral — better than guessing wrong)
    wrong answer    → -0.5  (penalizes hallucination)
    no \\boxed{}    →  0.0  (treated as abstention)

To use:
    Set IF_LLM_JUDGE_URL to an OpenAI-compatible /v1/chat/completions endpoint in the sbatch script.
"""

import logging
import os
import re

import requests

from .utils import timeout_limit, parse_judge_yes_no, pick_judge_url

logger = logging.getLogger(__name__)

IDK_TOKEN = "[IDK]"
BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")

CORRECT_SCORE = 1.0
ABSTAIN_SCORE = 0.0
WRONG_SCORE = -0.5
NO_ANSWER_SCORE = 0.0

JUDGE_MODEL = os.getenv("IF_LLM_JUDGE_MODEL", "gpt-oss-20b")
JUDGE_TIMEOUT = 240

IF_JUDGE_PROMPT_TEMPLATE = (
    "### Question: {question}\n\n"
    "### Ground Truth Answer: {reference}\n\n"
    "### Student Answer: {student}\n\n"
    "For the above question, please verify if the student's answer is equivalent "
    "to the ground truth answer.\n"
    "Do not solve the question yourself; just check if the student's answer is "
    "equivalent to the ground truth answer.\n"
    "Do not explain your reasoning.\n"
    "Output exactly one line:\n"
    "Final Decision: Yes\n"
    "or\n"
    "Final Decision: No"
)


def _extract_last_boxed(text: str) -> str | None:
    matches = list(BOXED_PATTERN.finditer(text))
    return matches[-1].group(1).strip() if matches else None

# To do: unify `_llm_judge` functions to take `prompt` as input and be shared across different reward modules that need LLM judging, e.g. llm_judge_stem.py
def _llm_judge(question: str, student: str, reference: str) -> float:
    url = pick_judge_url().rstrip("/") + "/v1/chat/completions"
    prompt = IF_JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference, student=student
    )
    resp = requests.post(
        url,
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        timeout=JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    logger.info("Judge response: %r", text)
    if text is None:
        raise ValueError("Judge returned None response")
    return float(parse_judge_yes_no(text))


def _compute_score_impl(solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    extracted = _extract_last_boxed(str(solution_str))

    if extracted is None:
        return {"score": NO_ANSWER_SCORE, "acc": 0.0, "abstained": True}

    if extracted.strip().upper() == IDK_TOKEN:
        return {"score": ABSTAIN_SCORE, "acc": 0.0, "abstained": True}

    if not extra_info or not extra_info.get("question"):
        raise ValueError("extra_info must contain a non-empty 'question' field for llm judge prompt")
    question = extra_info["question"]
    try:
        is_correct = _llm_judge(question=question, student=extracted, reference=ground_truth)
    except Exception:
        logger.exception("LLM judge failed for extracted=%r", extracted)
        return {"score": 0.0, "acc": 0.0, "abstained": False, "llm_judge_failed": True}

    return {
        "score": CORRECT_SCORE if is_correct == 1.0 else WRONG_SCORE,
        "acc": is_correct,
        "abstained": False,
    }


@timeout_limit(seconds=1200.0)
def _compute_score_with_timeout(solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info)


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    solution_str = str(solution_str)
    ground_truth = str(ground_truth)
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info or {})
    except TimeoutError:
        logger.exception("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0, "abstained": False, "llm_judge_failed": True}
    except Exception:
        logger.exception("compute_score failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0, "abstained": False, "llm_judge_failed": True}
