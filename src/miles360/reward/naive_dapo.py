# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import asyncio
import logging
import math
import re
from typing import Optional

import sympy
from sympy.parsing import sympy_parser
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

from .prime_grader_lib.prime_grader_utils import math_equal
from .llm_judge_lib.utils import normalize_math_answer
from .utils import timeout_limit


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    # latex2text intentionally omitted — pylatexenc not required
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """Automatically make a mixed number evalable, e.g. 7 3/4 => 7+3/4."""
    p1 = re.compile("([0-9]) +([0-9])")
    return p1.sub("\\1+\\2", step)


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> Optional[str]:
    """Normalize answer expressions."""
    if expr is None:
        return None

    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch",
        "yard", "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    return len(set(x for x in expr if x.isalpha()))


def should_allow_eval(expr: str) -> bool:
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    """Check symbolic equality via sympy.

    Note: no @timeout_limit decorator — the caller is expected to be wrapped in
    asyncio.wait_for so that a hung sympy call doesn't block the event loop.
    """
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """Split elements in a tuple/interval, handling well-formatted commas in large numbers."""
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: str, ground_truth: str) -> tuple[bool, str]:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False, ""

    # Use normalize_math_answer from miles360/reward/utils.py
    # (equivalent to math_normalize.normalize_answer from prime_math)
    ground_truth_normalized_mathd = normalize_math_answer(ground_truth)
    given_answer_normalized_mathd = normalize_math_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True, given_answer_normalized_mathd

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False, given_normalized

    if ground_truth_normalized == given_normalized:
        return True, given_normalized

    if len(given_normalized) == 0:
        return False, given_normalized

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                try:
                    is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                except Exception:
                    is_correct = False
            if not is_correct:
                break

    return is_correct, given_normalized


def _last_boxed_only_string(string: str) -> Optional[str]:
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

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response: str) -> tuple[bool, str]:
    is_matched = False
    response = response.split("</think>")[-1]

    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    return is_matched, response


def _compute_score_impl(solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    """Top-level (picklable) implementation of the naive_dapo scoring logic.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (common in CUDA / distributed training).
    """
    model_output = str(solution_str)
    ground_truth = str(ground_truth)

    is_matched, extracted_model_output = match_answer(model_output)

    correct, pred = grade_answer(extracted_model_output, ground_truth)

    if not correct:
        try:
            if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi))
                correct = any(equivs)
            else:
                correct = math_equal(extracted_model_output, ground_truth, timeout=True)
        except Exception:
            correct = False

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
    }


@timeout_limit(seconds=30.0)
def _compute_score_with_timeout(solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    return _compute_score_impl(solution_str, ground_truth, extra_info)


def compute_score(solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    """Compute the reward score for a solution.

    Draws from the LLM-as-judge and PRIME reward functions.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        extra_info: Additional info (currently unused)

    Returns:
        dict: {"score": float, "acc": bool}
            score is 1.0 for correct, 0.0 for incorrect
    """
    try:
        return _compute_score_with_timeout(solution_str, ground_truth, extra_info)
    except TimeoutError:
        logger.warning("compute_score timed out for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
    except Exception:
        logger.exception("_compute_score_impl failed for solution_str=%r", solution_str)
        return {"score": 0.0, "acc": False}
