"""
Reward scorer for nvidia/Nemotron-RL-InverseIFEval-v1.

Each record has 3-10 LLM judge checks stored in `label` as a JSON list.
Each check contains a self-contained grading prompt (`content`) and a
`pass_criteria` (YES or NO). The model response is appended to the prompt
and sent to the judge. 

Reward:
    computed as the fraction of checks where judge answer matches pass_criteria.

To use:
    Set IF_LLM_JUDGE_URL to an OpenAI-compatible /v1/chat/completions endpoint.
"""

import json
import logging
import os

import requests

from .utils import timeout_limit, parse_judge_yes_no, pick_judge_url

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.getenv("IF_LLM_JUDGE_MODEL", "gpt-oss-20b")
JUDGE_TIMEOUT = 240

INVERSEIF_JUDGE_PROMPT_TEMPLATE = (
    "{content}\n\n"
    "## Model Response\n{model_response}\n\n"
    "Do not explain your reasoning.\n"
    "Output exactly one line:\n"
    "Final Decision: Yes\n"
    "or\n"
    "Final Decision: No"
)

def _call_judge(content: str, model_response: str) -> str:
    """Send a grading rubric to the judge and return the raw YES/NO response text."""

    url = pick_judge_url().rstrip("/") + "/v1/chat/completions"

    prompt = INVERSEIF_JUDGE_PROMPT_TEMPLATE.format(content=content, model_response=model_response)

    resp = requests.post(
        url,
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        timeout=JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    logger.info("Judge response: %r", text)
    return text


def _compute_score_impl(solution_str: str, ground_truth: str) -> dict:
    # ground truth is json dump of a list of dictionaries of llm judge rubrics
    try:
        checks = json.loads(ground_truth)
    except json.JSONDecodeError:
        raise ValueError(f"label is not valid JSON: {ground_truth!r}")

    if not checks:
        raise ValueError("label contains no checks")

    passed = 0
    failed_calls = 0
    for check in checks:
        content = check["content"]
        pass_criteria = check["pass_criteria"].strip().upper()
        try:
            judge_reply = _call_judge(content, solution_str)
        except Exception:
            logger.exception("Judge call failed for uid=%s rubric=%r", check.get("uid"), content)
            failed_calls += 1
            continue
        if judge_reply is None:
            logger.warning("Judge returned None for uid=%s rubric=%r", check.get("uid"), content)
            failed_calls += 1
            continue
        judge_answer = "YES" if parse_judge_yes_no(judge_reply) else "NO"
        if judge_answer == pass_criteria:
            passed += 1

    score = passed / len(checks)
    return {"score": score, "acc": score, "failed_judge_calls": failed_calls, "llm_judge_failed": failed_calls > 0}


@timeout_limit(seconds=1200.0)
def _compute_score_with_timeout(solution_str: str, ground_truth: str) -> dict:
    return _compute_score_impl(solution_str, ground_truth)


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    solution_str = str(solution_str)
    ground_truth = str(ground_truth)
    try:
        return _compute_score_with_timeout(solution_str, ground_truth)
    except (ValueError, KeyError):
        raise
    except TimeoutError:
        logger.exception("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0, "failed_judge_calls": 0, "llm_judge_failed": True}
    except Exception:
        logger.exception("compute_score failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0, "failed_judge_calls": 0, "llm_judge_failed": True}
