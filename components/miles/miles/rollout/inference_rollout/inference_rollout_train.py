import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils import dumper_utils
from miles.utils.http_utils import get, post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _summarize_abort_response(response: object) -> tuple[int, int, list[str], list[str]]:
    """Flatten a harbor /abort_all response into
    (aborted_trials, already_done, aborted_instance_ids, already_done_instance_ids).

    The response is either a single-worker dict carrying ``aborted`` /
    ``already_done_items`` lists, or a router-aggregate dict whose per-worker dicts
    each carry their own lists under ``worker_results``."""
    if not isinstance(response, dict):
        logger.warning(f"[abort] abort_all returned non-dict response: {response!r}")
        return 0, 0, [], []
    worker_results = response.get("worker_results")
    workers = worker_results if isinstance(worker_results, list) else [response]
    aborted_trials = 0
    already_done = 0
    aborted_instances: list[str] = []
    already_done_instances: list[str] = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        aborted_trials += int(worker.get("aborted_trials", 0) or 0)
        already_done += int(worker.get("already_done", 0) or 0)
        for item in worker.get("aborted") or []:
            if isinstance(item, dict) and item.get("instance_id"):
                aborted_instances.append(str(item["instance_id"]))
        for item in worker.get("already_done_items") or []:
            if isinstance(item, dict) and item.get("instance_id"):
                already_done_instances.append(str(item["instance_id"]))
    return aborted_trials, already_done, aborted_instances, already_done_instances


async def _signal_harbor(harbor_url: str, rollout_id: int) -> None:
    """Tell the harbor server to cancel every in-flight trial and close its session.
    Raises on failure -- an abort we could not signal must not be silently dropped.

    Collects the /abort_all response and logs how many trials harbor cancelled and
    their instance ids, tagged with the rollout step that triggered the abort."""
    response = await post(f"{harbor_url}/abort_all", {})
    aborted_trials, already_done, aborted_instances, already_done_instances = _summarize_abort_response(response)
    logger.info(
        f"[abort] rollout_id={rollout_id} harbor abort_all: aborted_trials={aborted_trials} "
        f"already_done={already_done} aborted_instances={aborted_instances} "
        f"already_done_instances={already_done_instances}"
    )


async def _abort_all_engines(args: Namespace) -> None:
    urls = await get_worker_urls(args)
    logger.info(f"[abort] abort_all -> {urls}")
    results = await asyncio.gather(
        *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls],
        return_exceptions=True,
    )
    for url, r in zip(urls, results, strict=True):
        if isinstance(r, Exception):
            logger.warning(f"[abort] abort_all failed for {url}: {r!r}")


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    """End-of-step rollout abort.

    Agentic rollouts first signal the harbor server, which cancels each trial and
    closes its session. Every rollout then broadcasts abort_all to the engines to
    clear any in-flight decode, and the trainer drains the pending tasks.
    Partial-sample collection is preserved.
    """
    args = state.args
    assert not state.aborted
    state.aborted = True

    # How many rollout tasks are still in flight when the abort fires, and which
    # groups they are. The specific instances harbor cancels are reported by
    # /abort_all below.
    cancelled_names = sorted(task.get_name() for task in pendings)
    logger.info(f"[abort] rollout_id={rollout_id} draining {len(pendings)} in-flight rollout tasks: {cancelled_names}")

    is_agentic = bool(getattr(args, "use_session_server", False) and getattr(args, "custom_agent_function_path", None))
    if is_agentic:
        harbor_url = getattr(args, "agent_server_url", None)
        if not harbor_url:
            raise RuntimeError("agentic rollout abort requires --agent-server-url")
        await _signal_harbor(harbor_url, rollout_id)

    await _abort_all_engines(args)

    # Drain the still-pending tasks. For partial rollout, keep each drained group
    # that has a response, stamping its origin step if not already set.
    aborted_samples: list[list[Sample]] = []
    for task in asyncio.as_completed(pendings):
        try:
            group = await task
        except Exception as exc:  # a failed pending task must not abort the drain
            logger.error(f"[abort] pending rollout task raised: {exc!r}", exc_info=exc)
            continue
        if not args.partial_rollout or group is None:
            continue
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    if args.partial_rollout:
        logger.info(f"[abort] collected {sum(len(x) for x in aborted_samples)} partial samples")

    return aborted_samples


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        return response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        return [worker["url"] for worker in response["workers"]]


def stamp_rollout_id(samples: list[list[Sample]], rollout_id: int) -> None:
    """Stamp the dispatching rollout id into every sample's metadata.

    sample.metadata flows through the agent function into the harbor /run
    payload, so trial records carry the step that executed them (a
    partial-rollout sample re-dispatched later is re-stamped; its origin
    stays in metadata["start_rollout_id"]).
    """
    for group in samples:
        for sample in group:
            sample.metadata["rollout_id"] = rollout_id


def submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]):
    tasks = []
    for group in samples:
        first = group[0][0] if isinstance(group[0], list) else group[0]
        tasks.append(
            asyncio.create_task(
                # submit a group of samples as a single task.
                generate_and_rm_group(
                    state,
                    group,
                    sampling_params=state.sampling_params.copy(),
                    evaluation=False,
                ),
                name=f"group-{first.index}",
            )
        )
    return tasks


async def generate_rollout_async(
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)
    if dynamic_filter is not None:
        logger.info(
            "Dynamic sampling filter: %s | min_reward_std=%s | min_mean_reward=%s | max_mean_reward=%s",
            args.dynamic_sampling_filter_path,
            getattr(args, "dynamic_sampling_min_reward_std", None),
            getattr(args, "dynamic_sampling_min_mean_reward", None),
            getattr(args, "dynamic_sampling_max_mean_reward", None),
        )
    if args.rollout_sample_filter_path:
        logger.info("Rollout sample filter: %s", args.rollout_sample_filter_path)

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    pendings = set()
    data = []
    all_data = []
    submitted = 0
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            if args.disable_oversampling and submitted >= target_data_size:
                break

            # get samples from the buffer and submit the generation requests.
            remaining = target_data_size - submitted
            n = remaining if args.disable_oversampling else args.over_sampling_batch_size
            if args.rolling_start_size:
                # Cap each wave to ~rolling_start_size rollouts (a group is
                # n_samples_per_prompt rollouts) so we don't open every session at once.
                n = min(n, max(1, args.rolling_start_size // args.n_samples_per_prompt))
            samples = data_source(n)
            stamp_rollout_id(samples, rollout_id)
            submitted += len(samples)
            pendings.update(submit_generate_tasks(state, samples))
            if args.rolling_start_size and len(data) + len(pendings) < target_data_size:
                await asyncio.sleep(args.rolling_start_interval)

        if not pendings:
            break

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

        groups_left = (target_data_size - submitted) + len(pendings)
        if args.disable_oversampling and 0 < groups_left <= args.tail_cancel_groups:
            logger.info(
                f"[rollout] tail-cancel: {groups_left} groups left "
                f"({len(pendings)} in flight, {target_data_size - submitted} unsubmitted) <= "
                f"{args.tail_cancel_groups}; cutting tail with {len(data)}/{target_data_size} "
                f"groups collected"
            )
            break

    pbar.close()
    if data:
        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
        )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(state, pendings, rollout_id)

    if args.disable_oversampling:
        if len(data) < args.rollout_batch_size:
            logger.warning(
                f"[rollout] oversampling disabled: {len(data)}/{args.rollout_batch_size} groups survived the dynamic filter"
            )
    else:
        assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    # There can be circumstances where users want to process all samples including filtered ones.
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
