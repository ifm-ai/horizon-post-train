"""Measure task difficulty with the exact initial policy and no optimizer steps."""

import asyncio
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def summarize_outcomes(outcomes, attempts=16, minimum=0.2, maximum=0.4):
    if attempts <= 0 or not 0 <= minimum <= maximum <= 1:
        raise ValueError("Invalid calibration sample count or pass-rate bounds")
    by_task = defaultdict(list)
    seen = set()
    for outcome in outcomes:
        identity = outcome["sample_index"]
        if identity in seen:
            raise ValueError(f"Duplicate calibration sample: {identity}")
        seen.add(identity)
        by_task[outcome["instance_id"]].append(outcome)
    summaries = []
    for instance_id, rows in sorted(by_task.items()):
        valid = [row for row in rows if not row["infra_failure"]]
        successes = sum(row["passed"] for row in valid)
        pass_rate = successes / len(valid) if valid else None
        complete = len(rows) == attempts and len(valid) == attempts
        summaries.append(
            {
                "instance_id": instance_id,
                "attempts": len(rows),
                "valid_attempts": len(valid),
                "infra_failures": len(rows) - len(valid),
                "successes": successes,
                "pass_at_1": pass_rate,
                "eligible": complete and minimum <= pass_rate <= maximum,
            }
        )
    return summaries


def write_task_dataset(source, records, output, selection):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite dataset: {output}")
    identities = [record["metadata"]["instance_id"] for record in records]
    if not records or len(identities) != len(set(identities)):
        raise ValueError("Dataset must contain unique tasks")
    for record in records:
        if not (source / record["metadata"]["instance_id"]).is_dir():
            raise ValueError("Missing source task directory")
    output.mkdir(parents=True)
    for instance_id in identities:
        (output / instance_id).symlink_to(
            (source / instance_id).resolve(), target_is_directory=True
        )
    (output / "harbor_records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")


def select_calibrated_dataset(
    source,
    summaries,
    output,
    provenance,
    count=32,
    minimum=0.2,
    maximum=0.4,
    attempts=16,
    seed=43,
):
    source = Path(source)
    records = [
        json.loads(line)
        for line in (source / "harbor_records.jsonl").read_text().splitlines()
    ]
    by_id = {record["metadata"]["instance_id"]: record for record in records}
    if len(by_id) != len(summaries) or set(by_id) != {
        row["instance_id"] for row in summaries
    }:
        raise ValueError("Calibration did not cover the exact candidate dataset")
    if any(row["attempts"] != attempts for row in summaries):
        raise ValueError(
            "Calibration did not collect the requested attempts for every task"
        )
    eligible = [
        row
        for row in summaries
        if row["valid_attempts"] == attempts and minimum <= row["pass_at_1"] <= maximum
    ]
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} tasks meet {minimum:.0%}–{maximum:.0%} with {attempts} valid attempts; need {count}. Refusing to widen the range or launch training."
        )
    chosen = random.Random(seed).sample(eligible, count)
    selection = {
        "source": str(source.resolve()),
        "source_selection": str(source / "selection.json"),
        "sampling_seed": seed,
        "domain_counts": dict(
            Counter(by_id[row["instance_id"]]["metadata"]["tags"][0] for row in chosen)
        ),
        "pass_at_1_range": [minimum, maximum],
        "measurement": "fresh frozen-policy pass rate, raw verifier reward >= 0.5, before training filters",
        "calibration": provenance,
        "tasks": chosen,
    }
    write_task_dataset(
        source, [by_id[row["instance_id"]] for row in chosen], output, selection
    )
    return chosen


def sample_outcome(sample, infra_failure):
    metadata = sample.metadata or {}
    reward = metadata.get("reward")
    numeric = (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(reward)
    )
    invalid = infra_failure or not numeric
    return {
        "instance_id": metadata["instance_id"],
        "sample_index": sample.index,
        "reward": reward if numeric else None,
        "passed": bool(not invalid and reward >= 0.5),
        "infra_failure": bool(invalid),
        "exit_status": metadata.get("exit_status"),
    }


def keep_all_groups(args, samples, **kwargs):
    from miles.rollout.filter_hub.base_types import DynamicFilterOutput

    return DynamicFilterOutput(keep=True)


def record_all_samples(args, all_samples, data_source=None):
    from miles.rollout.filter_hub.dynamic_sampling_filters import (
        _flatten_samples,
        is_infra_failure,
    )

    from agent360.harbor.miles.generate import log_all_samples

    samples = list(_flatten_samples(all_samples))
    outcomes = [sample_outcome(sample, is_infra_failure(sample)) for sample in samples]
    with Path(args.calibration_results).open("a") as stream:
        stream.write("".join(json.dumps(outcome) + "\n" for outcome in outcomes))
    log_all_samples(args, all_samples, data_source)


async def run_frozen_rollouts(args, rollout_manager, actor_model, on_batch=None):
    await actor_model.update_weights()
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data = await rollout_manager.generate.remote(rollout_id)
        del rollout_data
        print(
            f"Frozen calibration batch {rollout_id + 1}/{args.num_rollout} complete; optimizer steps=0",
            flush=True,
        )
        if on_batch is not None and on_batch(rollout_id):
            break


def validate_calibration_args(args):
    if args.debug_rollout_only or args.colocate or args.use_critic:
        raise ValueError(
            "Calibration requires the standard actor checkpoint loader and separate rollout engines"
        )
    if args.n_samples_per_prompt != 16 or args.rollout_batch_size != 32:
        raise ValueError("Expected 32 tasks per batch and 16 attempts per task")
    if (
        args.dynamic_sampling_filter_path
        != "agent360.harbor.miles.calibration.keep_all_groups"
        or args.rollout_all_samples_process_path
        != "agent360.harbor.miles.calibration.record_all_samples"
    ):
        raise ValueError("Calibration must retain and record every generated group")
    if not args.disable_oversampling or args.tail_cancel_groups != 0:
        raise ValueError("Calibration must not oversample or cancel slow task groups")
    agent_config = getattr(args, "custom_agent_config", None)
    if not isinstance(agent_config, dict):
        raise TypeError(
            "Calibration requires --custom-agent-config-json as a parsed object"
        )
    agent_overrides = agent_config.get("agent_overrides", {})
    format_penalty = (
        agent_overrides.get("format_error_reward_penalty")
        if isinstance(agent_overrides, dict)
        else None
    )
    if not isinstance(format_penalty, dict) or format_penalty.get("value") != 0:
        raise ValueError(
            "Calibration must measure raw verifier scores without format penalties"
        )


async def calibrate(args):
    validate_calibration_args(args)

    from miles.ray.placement_group import (
        create_placement_groups,
        create_rollout_manager,
        create_training_models,
    )
    from miles.utils.logging_utils import configure_logger
    from miles.utils.tracking_utils import init_tracking

    snapshot = Path(os.environ["SNAP_DIR"])
    args.calibration_results = str(snapshot / "calibration_samples.jsonl")
    if Path(args.calibration_results).exists():
        raise ValueError("Refusing to append to an existing calibration run")
    source = Path(args.prompt_data).parent
    records = [
        json.loads(line) for line in Path(args.prompt_data).read_text().splitlines()
    ]
    if len(records) != args.num_rollout * args.rollout_batch_size:
        raise ValueError(
            "Calibration must visit every candidate exactly once without dataset wrapping"
        )
    provenance = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hf_checkpoint": args.hf_checkpoint,
        "actor_load": args.load,
        "reference_load": args.ref_load,
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "attempts_per_task": args.n_samples_per_prompt,
        "success_threshold": 0.5,
        "format_reward_penalty": 0,
        "source_selection": str(source / "selection.json"),
        "candidate_count": len(records),
        "optimizer_steps": 0,
        "initial_weight_syncs": 1,
        "summary": str(snapshot / "calibration_summary.json"),
    }
    (snapshot / "calibration_config.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=256))
    configure_logger()
    placement_groups = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, _ = create_rollout_manager(args, placement_groups["rollout"])

    def report_progress(rollout_id):
        outcomes = [
            json.loads(line)
            for line in Path(args.calibration_results).read_text().splitlines()
        ]
        summaries = summarize_outcomes(outcomes)
        (snapshot / "calibration_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        eligible = sum(row["eligible"] for row in summaries)
        print(
            f"Calibration progress: {len(summaries)} tasks measured, {eligible} in the 20–40% band",
            flush=True,
        )
        return eligible >= 32

    try:
        actor_model, _ = await create_training_models(
            args, placement_groups, rollout_manager
        )
        if args.start_rollout_id != 0:
            raise ValueError(
                "Calibration must start from the original checkpoint, not a trained rollout"
            )
        await run_frozen_rollouts(args, rollout_manager, actor_model, report_progress)
    finally:
        await rollout_manager.dispose.remote()
    outcomes = [
        json.loads(line)
        for line in Path(args.calibration_results).read_text().splitlines()
    ]
    summaries = summarize_outcomes(outcomes)
    (snapshot / "calibration_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    measured_ids = {row["instance_id"] for row in summaries}
    measured_records = [
        record
        for record in records
        if record["metadata"]["instance_id"] in measured_ids
    ]
    if len(measured_records) != len(measured_ids):
        raise ValueError("Calibration returned an unknown task")
    provenance["measured_task_count"] = len(measured_records)
    (snapshot / "calibration_config.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    measured_source = snapshot / "measured_tasks"
    write_task_dataset(
        source,
        measured_records,
        measured_source,
        {"source_selection": str(source / "selection.json"), "calibration": provenance},
    )
    chosen = select_calibrated_dataset(
        measured_source, summaries, snapshot / "selected_tasks", provenance
    )
    print(
        f"Selected {len(chosen)} freshly calibrated tasks in the 20–40% band; ready for a new training run",
        flush=True,
    )


if __name__ == "__main__":
    from miles.utils.arguments import parse_args

    asyncio.run(calibrate(parse_args()))
