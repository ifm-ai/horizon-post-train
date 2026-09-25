"""
Reward scorer for nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1.

Each record contains a multi-turn conversation and 3-8 rubric checks stored
in `label` as a JSON list. Each check has a `question` (YES/NO behavioral
question) and a `pass_criteria` (YES or NO).

The judge receives the full conversation history, the model response, the
ground truth reference answer, and the rubric question.

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

MULTITURN_IF_JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating a model's response in a multi-turn conversation.\n\n"
    "## Conversation History\n{conversation}\n\n"
    "## Model Response\n{model_response}\n\n"
    "## Reference Answer\n{ground_truth_answer}\n\n"
    "## Evaluation Question\n{question}\n\n"
    "Please evaluate the model's response based on the conversation history and the reference answer.\n"
    "Answer the evaluation question above.\n"
    "Do not explain your reasoning.\n"
    "Output exactly one line:\n"
    "Final Decision: Yes\n"
    "or\n"
    "Final Decision: No"
)


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t["role"].upper()
        lines.append(f"[{role}]: {t['content']}")
    return "\n\n".join(lines)


def _call_judge(conversation: str, model_response: str, ground_truth_answer: str, question: str) -> str:
    """Send a grading rubric to the judge and return the raw YES/NO response text."""

    url = pick_judge_url().rstrip("/") + "/v1/chat/completions"

    prompt = MULTITURN_IF_JUDGE_PROMPT_TEMPLATE.format(
        conversation=conversation,
        model_response=model_response,
        ground_truth_answer=ground_truth_answer,
        question=question,
    )

    resp = requests.post(
        url,
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        timeout=JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    logger.info("Judge response: %r", text)
    return text


def _compute_score_impl(solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    # ground truth is json dump of a list of dictionaries of llm judge rubrics
    try:
        checks = json.loads(ground_truth)
    except json.JSONDecodeError:
        raise ValueError(f"label is not valid JSON: {ground_truth!r}")

    if not checks:
        raise ValueError("label contains no checks")

    if not extra_info or not extra_info.get("conversation"):
        raise ValueError("extra_info must be a non-empty dict containing a 'conversation' field")

    conversation = _format_conversation(extra_info["conversation"])
    ground_truth_answer = extra_info.get("ground_truth_answer", "")

    passed = 0
    failed_calls = 0
    for check in checks:
        question = check["question"]
        pass_criteria = check["pass_criteria"].strip().upper()
        try:
            judge_reply = _call_judge(
                conversation=conversation,
                model_response=solution_str,
                ground_truth_answer=ground_truth_answer,
                question=question,
            )
        except Exception:
            logger.exception("Judge call failed for question=%r", question)
            failed_calls += 1
            continue
        if judge_reply is None:
            logger.warning("Judge returned None for question=%r", question)
            failed_calls += 1
            continue
        judge_answer = "YES" if parse_judge_yes_no(judge_reply) else "NO"
        if judge_answer == pass_criteria:
            passed += 1

    score = passed / len(checks)
    return {"score": score, "acc": score, "failed_judge_calls": failed_calls, "llm_judge_failed": failed_calls > 0}


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
        return {"score": 0.0, "acc": 0.0, "failed_judge_calls": 0, "llm_judge_failed": True}
    except (ValueError, KeyError):
        raise
    except Exception:
        logger.exception("compute_score failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": 0.0, "failed_judge_calls": 0, "llm_judge_failed": True}
