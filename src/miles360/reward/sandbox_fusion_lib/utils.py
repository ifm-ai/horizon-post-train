import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Call the remote Sandbox Fusion API with retry logic."""
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "%sAttempt %s/%s: Calling sandbox API at %s",
                log_prefix,
                attempt + 1,
                MAX_RETRIES,
                sandbox_fusion_url,
            )
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
            )

            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) "
                    f"on attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info("%sRetrying after %s seconds...", log_prefix, delay)
                    time.sleep(delay)
                continue

            response.raise_for_status()

            logger.info("%sSandbox API call successful on attempt %s", log_prefix, attempt + 1)
            return response.json(), None

        except requests.exceptions.RequestException as exc:
            last_error = f"{log_prefix}API Request Error: {exc}"
            break
        except json.JSONDecodeError as exc:
            last_error = f"{log_prefix}API Response JSON Decode Error: {exc}"
            break
        except Exception as exc:
            last_error = f"{log_prefix}Unexpected Error: {exc}"
            break

    logger.error("%sSandbox API call failed. Last error: %s", log_prefix, last_error)
    if last_error:
        return None, last_error.replace(log_prefix, "API Call Failed: ")
    return None, "API Call Failed after retries"


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """Process a single test case against Sandbox Fusion."""
    api_response = None
    error_msg = None
    logger.info("Processing test case %s.", case_index + 1)

    current_generation_code = generation

    if fn_name and language == "python":
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip():
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(
                f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\n"
                f"Input was: {{_raw_input_str[:200]}}\\n"
            )
            return None, True

    try:
        _target_callable = None
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        elif 'Solution' in globals():
            _solution_instance = globals()['Solution']()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True

        _fn_result = _target_callable(*_args)
        return _fn_result, False
    except Exception:
        sys.stderr.write(
            f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n"
        )
        return None, True

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result))
        else:
            print(str(_result))
"""
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as exc:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {exc}"
        logger.error("Case %s: %s", case_index + 1, error_msg)
        traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1

    if error_msg:
        metadata["status"] = "api_error"
        logger.error("Case %s: API error occurred: %s", case_index, error_msg)
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        logger.error("Case %s: code: %s", case_index, generation_to_log)
        logger.error("Case %s: input: %s", case_index, stdin)
    elif api_response:
        logger.debug("Case %s: API Response: %s", case_index, api_response)
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        api_status = metadata["api_status"]

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            logger.debug("API returned Failed status. Response: %s", api_response)
            logger.debug("Compile Result: %s", compile_result)
            logger.debug("Run Result: %s", run_result)
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                metadata["status"] = (
                    "compile_timeout" if metadata["compile_status"] == "TimeLimitExceeded" else "compile_error"
                )
                result_status = -4
            elif run_result:
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"
                        result_status = -3
                    else:
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    logger.warning(
                        "Unknown run_status '%s' or state within Failed API status.",
                        metadata["run_status"],
                    )
                    metadata["status"] = "unknown_failure"
                    result_status = -1
            else:
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                metadata["status"] = "unknown_failure_state"
                result_status = -1
        elif api_status == "Success":
            if run_result and metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                if str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                metadata["status"] = "unexpected_success_state"
                result_status = -1
        else:
            logger.warning("Unknown API status received: %s", api_status)
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1
    else:
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error("Case %s: Unknown API state (no response and no error message).", case_index)
    return result_status, metadata


def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Check generated code against input/output test cases."""
    logger.info("Starting correctness check for generation.")

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    results = [None] * num_cases
    metadata_list = [None] * num_cases

    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning(
            "Mismatch between number of inputs (%s) and outputs (%s).",
            len(inputs),
            len(expected_outputs),
        )
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    first_compile_error_index = -1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation,
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata

                if result_status == -4:
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        first_compile_error_index = index

            except Exception as exc:
                logger.error("Test case %s generated an exception: %s", index, exc)
                traceback.print_exc()
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]),
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    if first_compile_error_index != -1:
        logger.warning(
            "Compile error detected in case %s. Marking subsequent cases as compile errors.",
            first_compile_error_index,
        )
        for i in range(first_compile_error_index + 1, num_cases):
            if results[i] != -4:
                results[i] = -4
                if metadata_list[i] is None:
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]),
                        "api_request_error": None,
                        "status": "compile_error_skipped",
                    }
                else:
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info("Correctness check finished. Results: %s", results)
    return results, metadata_list
