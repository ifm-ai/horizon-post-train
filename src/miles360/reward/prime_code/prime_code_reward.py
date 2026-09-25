# Copyright 2024 PRIME team and/or its affiliates
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

import json
import logging
import traceback

import multiprocessing
import os
import sys
import traceback
from typing import Optional
import asyncio
from .testing_util import run_test
from miles.utils.types import Sample
# Configure logging
logger = logging.getLogger(__name__)
    
def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list


## the main `compute_score` function
def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion: if it is already a code block, then just use it, otherwise try to extract code block from the completion
    solution = completion.split("```python")[-1].split("```")[0]
    solution = solution.strip()
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            logger.error(f"Error parsing test_cases: {e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            ## Comment by Jalaj: to me this original logic was not correct, in binary mode, we should just return True or False irrespective of if all tests were passing
            # if success:
            #     return success, metadata

            ## In binary mode, just return True or False 
            if not continuous:
                return success, metadata
            # In continuous mode, if all tests passed, no need for per-sample breakdown
            if success:
                return success, metadata
                
        except Exception as e:
            logger.debug(f"Complete test failed: {e}")
            # the second except block returns False, None

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})
            
        # per sample test: if continuous score is needed, test first 10 samples regardless of failures
        # do not test all samples cuz some problems have enormous test cases
        metadata_list = []
        res_list = []
        for test_case_id, test_case in enumerate(test_cases_list):
            res, metadata = check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
            try:
                metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
            except Exception:
                metadata = {}
            metadata["test_case"] = {}
            metadata["test_case"]["input"] = str(test_case["inputs"][0])
            metadata["test_case"]["output"] = str(test_case["outputs"][0])
            metadata["test_case"]["res"] = str(res)
            metadata_list.append(metadata)
            res_list.extend(res)

            if test_case_id >= 9:
                break
        res_count = len(res_list) if len(res_list) > 0 else 1
        success = sum(map(lambda x: x is True, res_list)) / res_count
        
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None

    return success, metadata_list


'''
PRIME code reward function for miles.

Async and batched async functions which wrap the `compute_score` function above

Usage:
    --custom-rm-path miles360.reward.prime_code:prime_code_reward:prime_code_rm

Or for batch mode with --group-rm:
    --custom-rm-path miles360.reward.prime_code:prime_code_reward:batched_prime_code_rm
'''

async def prime_code_rm(args, sample: Sample, **kwargs) -> float:
    """Async reward function for miles.

    Compatible with miles' --custom-rm-path interface.
    Runs CPU-bound sympy operations in a thread pool to avoid blocking the event loop.

    Args:
        args: Namespace with training arguments
        sample: Sample object with prompt, response, label, metadata

    Returns:
        Reward score (float)
    """
    # Run CPU-bound computation in thread pool
    result = await asyncio.to_thread(
        compute_score,
        response=sample.response,
        test_cases=sample.label, # for code datasets, this is expected to be test cases
    )
    return result["success"]


async def batched_prime_code_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    """Batched async reward function for miles.

    Compatible with miles' --custom-rm-path interface when using --group-rm.
    Runs all samples concurrently using asyncio.gather.

    Args:
        args: Namespace with training arguments
        samples: List of Sample objects

    Returns:
        List of reward scores
    """
    # Run all samples concurrently in thread pool
    tasks = [
        asyncio.to_thread(
            compute_score,
            response=sample.response,
            test_cases=sample.label, # for code datasets, this is expected to be test cases
        )
        for sample in samples
    ]
    results = await asyncio.gather(*tasks)
    return [r["success"] for r in results]