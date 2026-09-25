# Modified for release: configurable deployment paths and public source references.
# Copyright 2024 PRIME team and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 OpenAI
# Copyright (c) 2021 Dan Hendrycks
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
"""
PRIME math grader.

This module provides comprehensive math equality checking with support for:
- Numerical equality with tolerance
- Symbolic equality via sympy
- Pi substitution
- Interval/tuple/matrix comparison

Reuses miles' math_utils where possible.
"""

import contextlib
import math
import re
from math import isclose
from typing import Optional

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


def _last_boxed_only_string(string: str) -> str | None:
    """Extract content from the last \\boxed{} or \\fbox{} in a string.

    Handles nested braces correctly.

    Args:
        string: Input string containing \\boxed{} or \\fbox{}

    Returns:
        Content inside the braces, or None if not found
    """
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
    """Extract the answer from a model response using multiple patterns.

    Handles:
    - "answer:", "answer is", "answers are" markers
    - "is answer", "is the answer" markers
    - \\boxed{} extraction
    - Sentence ending cleanup
    - "be ", "is ", "are ", "=" markers

    Args:
        response: The model's response text

    Returns:
        Tuple of (is_matched, extracted_answer)
    """
    is_matched = False

    # Check for "answer:", "answer is", "answers are" patterns
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()

    # Check for "is answer", "is the answer" patterns (answer comes before marker)
    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()

    # Find boxed content
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Clean up sentence endings
    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    # Check for additional markers
    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()

    # Answer must have a digit to be considered valid
    is_matched = is_matched if any(c.isdigit() for c in response) else False

    return is_matched, response

def is_digit(s) -> tuple[bool, float | None]:
    """Check if a string represents a number.

    Handles comma-formatted numbers like "1,000" and "{,}" notation.

    Args:
        s: Input value to check

    Returns:
        Tuple of (is_number, parsed_value)
    """
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None

def handle_base(x) -> str | int:
    """Handle base notation like '123_10' (123 in base 10).

    Args:
        x: Input value, possibly with base suffix

    Returns:
        Parsed integer if base notation found, otherwise original value
    """
    if isinstance(x, str) and "_" in x:
        ## Old code which does not seem correct except for base 10 ##
        # Due to base notation
        # x = x.split("_")[0]
        # x = float(x)
        # return int(x)
        parts = x.split("_")
        if len(parts) == 2:
            try:
                number_part = parts[0]
                base_part = int(parts[1])
                # use the built-in int function with base
                return int(number_part, base=base_part)
            except ValueError:
                # return original if conversion fails
                pass
    return x

def handle_pi(string, pi: float) -> str | float:
    """Substitute \\pi with a numeric value.

    Args:
        string: Input string containing \\pi
        pi: Numeric value to substitute (e.g., math.pi or 3.14)

    Returns:
        Evaluated expression or original string
    """
    if isinstance(string, str) and "\\pi" in string:
        # Find the first occurrence of "\\pi"
        idx = string.find("\\pi")

        # Iterate over the string and find all occurrences of "\\pi"
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\\pi" with "*pi" if the previous character is a digit
                string = string[:idx] + f"*{pi}" + string[idx + 3:]
            else:
                # Replace "\\pi" with "1*pi" if the previous character is not a digit
                string = string[:idx] + f"1*{pi}" + string[idx + 3:]

            # Find the next occurrence of "\\pi"
            idx = string.find("\\pi", idx + 1)

        # Evaluate the expression using eval() function
        with contextlib.suppress(Exception):
            string = eval(string)

    return string

def normalize(answer, pi: Optional[float] = None) -> str | int | float:
    """Normalize an answer for comparison.

    Handles:
    - Dollar amounts ($100)
    - Percentages (50% or 50\\%)
    - Base notation (123_10)
    - Pi substitution

    Args:
        answer: The answer to normalize
        pi: Value to use for pi substitution, defaults to math.pi

    Returns:
        Normalized answer string
    """
    # Check if answer is $<number> and remove $ to compare
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # Check if answer is <number>% or <number>\\% and remove %
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    # Handle base notation
    answer = handle_base(answer)

    # Handle pi
    if pi is None:
        print("INFO: Using default pi value (math.pi), pass pi parameter to override")
        pi = math.pi
    answer = handle_pi(answer, pi)

    return answer

def format_intervals(prediction: str) -> str:
    """Convert sympy Interval notation to bracket notation.

    Examples:
        Interval(a, b) -> [a, b]
        Interval.Ropen(a, b) -> [a, b)
        Interval.Lopen(a, b) -> (a, b]
        Interval.open(a, b) -> (a, b)

    Args:
        prediction: String potentially containing Interval notation

    Returns:
        String with bracket notation
    """
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Interval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Interval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Interval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Interval.open(a, b) == (a, b)
                return f"({inner_content})"

    return prediction

_LATEXISH_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _latexish_to_expr(s):
    """Antlr-free fallback for LaTeX-ish math strings.

    sympy's ``parse_latex`` needs ``antlr4-python3-runtime==4.11``, but our image
    pins 4.9.3 (omegaconf), so ``parse_latex`` raises ImportError there and
    ``symbolic_equal`` loses its LaTeX path. This regex-normalizes
    common LaTeX (``\\frac``, ``\\sqrt``, ``\\cdot`` ...) into sympy syntax and
    parses it with implicit multiplication -- no antlr required. It runs only after
    ``parse_expr`` and ``parse_latex`` have both failed, so it changes nothing where
    ``parse_latex`` works.
    """
    t = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\\[,;:!]", "", t)
    t = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", t)
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
    t = t.replace("{", "(").replace("}", ")")
    return parse_expr(t, transformations=_LATEXISH_TRANSFORMS)


def symbolic_equal(a, b, tolerance: float) -> bool:
    """Check symbolic equality using sympy.

    Tries multiple parsing strategies:
    1. parse_expr (Python-like expressions)
    2. parse_latex (LaTeX expressions)

    Then checks:
    1. simplify(a - b) == 0
    2. Numerical evaluation with tolerance

    Args:
        a: First expression
        b: Second expression
        tolerance: Relative tolerance for numerical comparison

    Returns:
        True if expressions are equal
    """
    def _parse(s):
        for f in [parse_expr, parse_latex, _latexish_to_expr]:
            try:
                return f(s)
            except Exception:
                continue
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=tolerance):
            return True
    except Exception:
        pass

    return False


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    pi: float = math.pi,
) -> bool:
    """Comprehensive math equality check.

    Checks equality through multiple methods:
    1. String comparison (case-insensitive, whitespace-normalized)
    2. Numerical equality with tolerance (handles percentages)
    3. Tuple/interval element-wise comparison
    4. Point and matrix comparison
    5. Symbolic equality via sympy

    Args:
        prediction: The predicted answer
        reference: The ground truth answer
        include_percentage: Whether to check 100x and /100 variants
        tolerance: Relative tolerance for numerical comparison
        pi: Value to use for pi substitution (default: math.pi)

    Returns:
        True if prediction matches reference
    """
    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    # Handle very long predictions (corner case)
    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    # 0. String comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    # 1. Numerical equality
    try:
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction_val = is_digit(prediction)[1]
            reference_val = is_digit(reference)[1]
            # Check with percentage variants
            gt_result = [reference_val / 100, reference_val, reference_val * 100] if include_percentage else [reference_val]
            for item in gt_result:
                try:
                    if isclose(item, prediction_val, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. Symbolic equality
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    # [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_pt, ref_pt, include_percentage, tolerance, pi)
            for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=True)
        ):
            return True

    # Comma-separated values comparison
    if "," in prediction and "," in reference:
        # Skip if either starts with a bracket (let earlier blocks or symbolic handle it)
        # - Think of the cases: [a, b, c] vs (a, b, c) which are semantically different: 
        #   instead of recursive calls to compare '[a]' with '(a)', it will just skip to symbolic comparison
        if not (prediction[0] in "([{" or reference[0] in "([{"):
            pred_parts = [item.strip() for item in prediction.split(",")]
            ref_parts = [item.strip() for item in reference.split(",")]

            if len(pred_parts) == len(ref_parts):
                if all(
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance, pi)
                    for i in range(len(pred_parts))
                ):
                    return True

    # Point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1: -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_pt, ref_pt, include_percentage, tolerance, pi)
            for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=False)
        ):
            return True

    # Matrix comparison
    if "\\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                math_equal(pred, ref, include_percentage, tolerance, pi)
                for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
            ):
                return True
        except Exception:
            pass
    elif "\\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        try:
            pred_matrix = eval(prediction)
            if isinstance(pred_matrix, list):
                ref_matrix_items = (
                    reference.lstrip("\\begin{pmatrix}")
                    .lstrip("\\begin{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                )
                ref_matrix_items = ref_matrix_items.split("\\")
                ref_matrix_items = [row.split("&") if "&" in row else row for row in ref_matrix_items]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    math_equal(pred, ref, include_percentage, tolerance, pi)
                    for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
                ):
                    return True
        except Exception:
            pass

    return symbolic_equal(prediction, reference, tolerance)
