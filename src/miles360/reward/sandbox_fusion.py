"""
Sandbox Fusion reward for miles.
"""

import json
import logging
import os
import traceback

from .sandbox_fusion_lib.utils import check_correctness

logger = logging.getLogger(__name__)


def compute_score(
    sandbox_fusion_url,
    concurrent_semaphore,
    memory_limit_mb,
    completion,
    test_cases,
    continuous=False,
    timeout=10,
):
    """
    Compute a code score using the remote Sandbox Fusion API.

    If `sandbox_fusion_url` is not passed, the function falls back to the
    `SANDBOX_FUSION_URL` environment variable.
    """
    if memory_limit_mb is None:
        memory_limit_mb = 1024

    url_base = sandbox_fusion_url or os.getenv("SANDBOX_FUSION_URL")
    if not url_base:
        raise ValueError("SANDBOX_FUSION_URL is not set")

    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse test_cases JSON: %s", exc)
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=url_base,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

        if not res_list:
            return 0.0, metadata_list

        if continuous:
            num_to_consider = min(len(res_list), 10)
            score = 0.0 if num_to_consider == 0 else sum(1 for r in res_list[:num_to_consider] if r is True) / num_to_consider
            final_metadata = metadata_list
        else:
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list

    except Exception as exc:
        logger.error("Error during compute_score: %s", exc)
        traceback.print_exc()
        score = 0.0
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {exc}"}]

    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
