import asyncio
import logging
import os
import weakref

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY = 4
_IF_LLM_JUDGE_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY = 4
_STEM_LLM_JUDGE_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_if_llm_judge_max_concurrency() -> int:
    raw = os.getenv("IF_LLM_JUDGE_MAX_CONCURRENCY", str(_DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY))
    try:
        limit = int(raw)
    except ValueError:
        logger.warning(
            "Invalid IF_LLM_JUDGE_MAX_CONCURRENCY=%r; using default %d",
            raw,
            _DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY,
        )
        return _DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY

    if limit < 1:
        logger.warning(
            "IF_LLM_JUDGE_MAX_CONCURRENCY must be >= 1; using default %d",
            _DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY,
        )
        return _DEFAULT_IF_LLM_JUDGE_MAX_CONCURRENCY
    return limit


def _get_if_llm_judge_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = _get_if_llm_judge_max_concurrency()
    cached = _IF_LLM_JUDGE_SEMAPHORES.get(loop)
    if cached is None or cached[0] != limit:
        semaphore = asyncio.Semaphore(limit)
        _IF_LLM_JUDGE_SEMAPHORES[loop] = (limit, semaphore)
        logger.info("IF LLM judge max concurrency set to %d", limit)
        return semaphore
    return cached[1]


def _get_stem_llm_judge_max_concurrency() -> int:
    raw = os.getenv("STEM_LLM_JUDGE_MAX_CONCURRENCY", str(_DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY))
    try:
        limit = int(raw)
    except ValueError:
        logger.warning(
            "Invalid STEM_LLM_JUDGE_MAX_CONCURRENCY=%r; using default %d",
            raw,
            _DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY,
        )
        return _DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY

    if limit < 1:
        logger.warning(
            "STEM_LLM_JUDGE_MAX_CONCURRENCY must be >= 1; using default %d",
            _DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY,
        )
        return _DEFAULT_STEM_LLM_JUDGE_MAX_CONCURRENCY
    return limit


def _get_stem_llm_judge_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = _get_stem_llm_judge_max_concurrency()
    cached = _STEM_LLM_JUDGE_SEMAPHORES.get(loop)
    if cached is None or cached[0] != limit:
        semaphore = asyncio.Semaphore(limit)
        _STEM_LLM_JUDGE_SEMAPHORES[loop] = (limit, semaphore)
        logger.info("STEM LLM judge max concurrency set to %d", limit)
        return semaphore
    return cached[1]


async def _run_if_llm_judge_compute(compute_score, solution_str: str, ground_truth: str, extra_info: dict):
    semaphore = _get_if_llm_judge_semaphore()
    async with semaphore:
        return await asyncio.to_thread(compute_score, solution_str, ground_truth, extra_info=extra_info)


async def _run_stem_llm_judge_compute(compute_score, *args, **kwargs):
    semaphore = _get_stem_llm_judge_semaphore()
    async with semaphore:
        return await asyncio.to_thread(compute_score, *args, **kwargs)


def _normalize_score(res) -> float:
    if isinstance(res, dict):
        return float(res.get("score", res.get("acc", 0.0)))
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


async def async_rm(args, sample: Sample, **kwargs) -> float:
    extra_info = getattr(sample, "metadata", None) or {}
    data_source = (extra_info.get("data_source") or "").strip()
    reward_metric = None
    if extra_info and isinstance(extra_info, dict):
        reward_metric = extra_info.get("reward_metric", None)

    solution_str = sample.response
    ground_truth = sample.label

    # math
    if data_source.startswith("math"):
        if reward_metric == "prime_math":
            from . import prime_math
            res = prime_math.compute_prime_math_reward(solution_str, ground_truth)
        elif reward_metric == "math_llm_judge":
            from . import llm_judge_math
            res = await asyncio.to_thread(llm_judge_math.compute_score, solution_str, ground_truth, extra_info)
        elif reward_metric == "math_dapo" or data_source.startswith("math_dapo"):
            from . import math_dapo
            res = math_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
        else:
            from . import naive_dapo
            res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # coding — subprocess/multiprocessing-based; run in a thread so the event loop
    # stays free for concurrent LLM judge calls in the same batch
    elif data_source.startswith('codegen'):
        from . import coder1
        res = await asyncio.to_thread(coder1.compute_score, solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("simulation__codeio"):
        from . import codeio
        res = await asyncio.to_thread(codeio.compute_score, solution_str, ground_truth)
    elif data_source.startswith("simulation__cruxeval"):
        from . import cruxeval
        res = await asyncio.to_thread(cruxeval.compute_score, solution_str, ground_truth)
    # simulation / logic
    elif data_source.startswith("simulation__arcagi") or data_source.startswith("simulation__barc"):
        from . import arcagi
        res = arcagi.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("logic__zebra_puzzle"):
        from . import zebra_puzzle
        res = await asyncio.to_thread(zebra_puzzle.compute_score, solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("logic__ordering_puzzle"):
        from . import logic_ordering_puzzle
        res = logic_ordering_puzzle.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("logic__graph"):
        from . import logic_graph
        res = logic_graph.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # tabluar reasoning
    elif data_source.startswith("table"):
        from . import tabular_reasoning
        res = tabular_reasoning.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # STEM
    elif data_source.startswith('stem__gpqa'):
        from . import gpqa
        from . import supergpqa
        if "no_box" in data_source:
            res = gpqa.compute_score(solution_str, ground_truth)
        else:
            res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem__supergpqa'):
        from . import supergpqa
        res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith("stem_web"):
        from . import llm_judge_stem
        res = await _run_stem_llm_judge_compute(llm_judge_stem.compute_score, solution_str, ground_truth, extra_info)
    # Reasoning gym
    elif data_source == "reasoning_gym":
        from . import reasoning_gym
        res = reasoning_gym.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # OOD
    elif data_source == "ood__ifeval":
        from . import ifeval
        res = ifeval.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source == "ood__livebench":
        from . import livebench
        res = livebench.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source == "ood__ifbench":
        from . import ifbench
        res = ifbench.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["deepmath", "DeepMath", "zwhe99/DeepMath-103K"]:
        from . import deep_math
        res = deep_math.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["stem_nemotron", "nemotron_stem"]:
        from . import nemotron_stem
        res = nemotron_stem.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source == "nvidia_mcqa":
        from . import nvidia_mcqa
        res = nvidia_mcqa.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ("nvidia_abstention", "nemotron_qa_abstention"):
        from . import nvidia_abstention
        res = await _run_if_llm_judge_compute(
            nvidia_abstention.compute_score,
            solution_str,
            ground_truth,
            extra_info,
        )
    elif data_source == "nvidia_inverse_ifeval":
        from . import nvidia_inverse_ifeval
        res = await _run_if_llm_judge_compute(
            nvidia_inverse_ifeval.compute_score,
            solution_str,
            ground_truth,
            extra_info,
        )
    elif data_source == "nvidia_multiturn_if":
        from . import nvidia_multiturn_if
        res = await _run_if_llm_judge_compute(
            nvidia_multiturn_if.compute_score,
            solution_str,
            ground_truth,
            extra_info,
        )
    elif data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    # named datasets
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math
        res = prime_math.compute_prime_math_reward(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        sandbox_fusion_url = os.getenv("SANDBOX_FUSION_URL")
        if sandbox_fusion_url:
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=None,
                memory_limit_mb=1024,
                completion=solution_str,
                test_cases=ground_truth,
                continuous=True,
            )
        else:
            from . import prime_code
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    # TODO: search_r1_like_qa_em assumes ground_truth has "target" field
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source.startswith("synlogic"):
        from . import synlogic
        res = synlogic.compute_score(solution_str, ground_truth, extra_info=extra_info, data_source=data_source)
    elif data_source.startswith("textbook_reasoning"):
        from . import llm_judge_stem
        res = await _run_stem_llm_judge_compute(
            llm_judge_stem.compute_score_megascience,
            model_output=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        # set the `llm_judge_failed` flag in sample.metadata
        # can be read by rollout filters, e.g. mask_truncated_and_llm_judge_failed in rollout_filters.py
        if res.get("llm_judge_failed"):
            sample.metadata["llm_judge_failed"] = True
        if res.get("failed_judge_calls"):
            logger.warning(
                "failed_judge_calls=%d for data_source=%r solution_str=%r",
                res["failed_judge_calls"],
                data_source,
                solution_str[:100],
            )

    return _normalize_score(res)


async def batched_async_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    return list(await asyncio.gather(*tasks))
