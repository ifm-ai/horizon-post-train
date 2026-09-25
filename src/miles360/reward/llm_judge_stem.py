# Modified for release: configurable deployment paths and public source references.
"""
Minimal STEM judge: every answer is graded by an external LLM.

Prerequisite:
  - Set env var STEM_LLM_JUDGE_URL to an OpenAI-compatible /v1/chat/completions.
  - Launch the service with your preferred model beforehand, e.g.

e.g. with TIGER-Lab/general-verifier (4096 ctx limit):
    vllm serve TIGER-Lab/general-verifier --host $NODE_IP --data-parallel-size 8
    export STEM_LLM_JUDGE_URL=http://$NODE_IP:8000

e.g. with Qwen3-4B (32k ctx, /no_think to disable thinking):
    vllm serve /path/to/Qwen3-4B --served-model-name Qwen3-4B --max-model-len 32768
    export STEM_LLM_JUDGE_URL=http://127.0.0.1:8000

The default judge for training runs is gpt-oss-20b.
"""

import logging
import os
import requests

from .utils import parse_judge_yes_no, pick_judge_url

logger = logging.getLogger(__name__)

# ------------ Judge configuration (change here to swap models) ---------------

JUDGE_MODEL = os.getenv("STEM_LLM_JUDGE_MODEL", "gpt-oss-20b")

JUDGE_SYSTEM_MESSAGE = (
    "You are a strict STEM answer-equivalence judge. "
    "Use the reference answer as authoritative and output only the requested final decision."
)

JUDGE_STEM_PROMPT_TEMPLATE = (
    "Judge whether the student answer is equivalent to the reference answer for the question.\n\n"
    "## Question\n{question}\n\n"
    "## Reference Answer\n{reference}\n\n"
    "## Student Answer\n{student}\n\n"
    "Decision rules:\n"
    "- Mark Yes if the student answer has the same final meaning as the reference answer.\n"
    "- Accept algebraically equivalent forms, equivalent units, common notation differences, and reasonable rounding.\n"
    "- For multiple-choice questions, accept either the correct option letter or the option text.\n"
    "- Ignore reasoning quality and formatting; judge only the final answer.\n"
    "- Mark No if the answer is missing, ambiguous, not comparable, or includes a contradictory final answer.\n"
    "- Do not give partial credit.\n\n"
    "Do not explain your reasoning.\n"
    "Output exactly one line:\n"
    "Final Decision: Yes\n"
    "or\n"
    "Final Decision: No"
)

LEGACY_JUDGE_QWEN_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {reference}\n\n"
    "### Student Answer: {student}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". "
    "If the student's answer is incorrect, output \"Final Decision: No\". /no_think Assistant:"
)

LEGACY_JUDGE_TIGERLAB_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {reference}\n\n"
    "### Student Answer: {student}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". "
    "If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

# Request timeout in seconds
JUDGE_TIMEOUT = 300


# ------------ Core LLM call --------------------------------------------------
def _llm_judge(question: str, student: str, reference: str, verbose: bool = False) -> float:
    url_base = pick_judge_url("STEM_LLM_JUDGE_URL")
    url = url_base.rstrip("/") + "/v1/chat/completions"

    prompt = JUDGE_STEM_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference,
        student=student,
    )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    resp = requests.post(url, json=payload, timeout=JUDGE_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    logger.info("Judge response: %r", text)
    if text is None:
        raise ValueError("Judge returned None response")
    score = float(parse_judge_yes_no(text))

    marker = "✅" if score == 1. else "❌"

    if verbose:
        print(marker*50)
        print("student answer: ", student)
        print("gt: ", reference)
        import json as _json
        print(_json.dumps(data, indent=2, ensure_ascii=False))
        print(marker*16 + " LLM Judge CONTENT " + marker*16)
        print(text)
        print(marker*16 + "End of LLM Judge Reply \n"+ marker*16)

    return score


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1:right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    response = response.split("</think>")[-1]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    return is_matched, response


# ------------ Public API -----------------------------------------------------
def compute_score(model_output: str,
                  ground_truth: str,
                  extra_info: dict) -> dict:
    """
    Arguments
    ---------
    model_output : str   – agent's raw answer
    ground_truth : str   – reference answer
    extra_info   : dict  – MUST contain key "question"

    Returns
    -------
    dict with keys:
        score            – 1.0 if correct, else 0.0
        acc              – bool correctness
        extracted_answer – the extracted student answer
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    is_matched, extracted_model_output = match_answer(model_output)
    question = extra_info["question"]
    if not is_matched:
        return {"score": 0.0, "acc": False, "extracted_answer": extracted_model_output}
    try:
        is_correct = bool(_llm_judge(question, extracted_model_output, ground_truth, verbose=False))
    except Exception:
        logger.exception("_llm_judge failed for extracted_answer=%r", extracted_model_output)
        return {"score": 0.0, "acc": False, "extracted_answer": extracted_model_output, "llm_judge_failed": True}
    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "extracted_answer": extracted_model_output,
    }

def compute_score_megascience(
                  model_output: str,
                  ground_truth: str,
                  extra_info: dict) -> dict:

    model_output = str(model_output)
    # if "</think>" in model_output:
    #     model_output = model_output.split("</think>")[-1].strip()
    # else:
    #     # truncated response — no </think> tag found, take last 2000 chars as best-effort answer
    #     # penalize the model for incorrect formatting
    #     model_output = model_output[-4096:].strip()

    ground_truth = str(ground_truth)
    question = extra_info["question"]

    try:
        is_correct = _llm_judge(question=question, student=model_output, reference=ground_truth, verbose=False)
    except Exception:
        logger.exception("_llm_judge failed for model_output=%r", model_output)
        return {"score": 0.0, "llm_judge_failed": True}

    return {"score": float(is_correct)}