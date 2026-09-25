"""Input checks and service coordination for coding-overfit32.sbatch.

Run ``python recipes/overfit_runtime.py --help`` for individual commands.
Input validation and HTTP checks use only Python's standard library. Container
and Ray checks import the training dependencies only when those commands run.
"""

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.request


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs():
    """Check shared paths, the 32-task manifest, and model argument syntax."""
    for key in ("RL360_ROOT", "RL360_HF_CHECKPOINT", "RL360_TORCH_CHECKPOINT",
                "RL360_TASKS_DIR", "RL360_MODEL_ARGS_FILE", "RL360_OUTPUT_DIR"):
        require(os.environ.get(key) and Path(os.environ[key]).is_absolute(),
                f"{key} must be an absolute shared path")
    hf = Path(os.environ["RL360_HF_CHECKPOINT"])
    checkpoint = Path(os.environ["RL360_TORCH_CHECKPOINT"])
    require((hf / "config.json").is_file(), "HF checkpoint needs config.json, tokenizer and weights")
    require((checkpoint / "latest_checkpointed_iteration.txt").is_file(),
            "TORCH checkpoint must be a converted Megatron checkpoint")
    tasks = Path(os.environ["RL360_TASKS_DIR"])
    rows = [json.loads(line) for line in (tasks / "harbor_records.jsonl").read_text().splitlines() if line.strip()]
    require(len(rows) == 32, "This recipe requires exactly 32 task records")
    identities = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("prompt"), (str, list)),
                "Each task needs a prompt string or message list")
        metadata = row.get("metadata")
        require(isinstance(metadata, dict), "Each task needs a metadata object")
        task_id = metadata.get("instance_id")
        require(isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id),
                "instance_id must be a safe, nonempty directory name")
        require(task_id not in identities, "Duplicate instance_id in task manifest")
        identities.add(task_id)
        # Calibration may link to task directories; mount their targets too.
        for filename in ("task.toml", "instruction.md", "tests/test.sh"):
            require((tasks / task_id / filename).is_file(), f"Task {task_id} is missing {filename}")

    model_args = json.loads(Path(os.environ["RL360_MODEL_ARGS_FILE"]).read_text())
    require(isinstance(model_args, list) and model_args and all(
        isinstance(arg, str) and arg and not any(c in arg for c in "\x00\n\r") for arg in model_args
    ), "Model args file must be a nonempty JSON array of argument strings")
    require(not any("<" in arg or ">" in arg for arg in model_args), "Replace placeholders in model args")
    for flag in ("--num-layers", "--hidden-size", "--num-attention-heads",
                 "--tito-model", "--sglang-tool-call-parser"):
        require(flag in model_args and model_args.index(flag) + 1 < len(model_args)
                and not model_args[model_args.index(flag) + 1].startswith("--"), f"Model args must supply {flag}")
    require(not any(arg.startswith("--wandb-key") for arg in model_args),
            "Pass W&B credentials through the environment")
    return model_args


def check_container(gpus_per_node):
    """Check source mounts, imports and visible GPUs inside the job container."""
    import torch

    for module in ("ray", "sglang", "smg.smg_rs", "harbor", "megatron.core", "miles", "agent360"):
        importlib.import_module(module)
    require(torch.cuda.device_count() == gpus_per_node, "GPU allocation mismatch")
    for key in ("RL360_HF_CHECKPOINT", "RL360_TORCH_CHECKPOINT", "RL360_TASKS_DIR"):
        require(Path(os.environ[key]).is_dir(), f"Missing mount: {key}")
    print("Container preflight passed", flush=True)


def wait_http(url, timeout):
    """Wait for a service's HTTP readiness endpoint without using a proxy."""
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Service did not become ready; check the run logs: {url}")


def wait_ray(nodes, gpus, timeout):
    """Wait until every allocated Ray worker has registered its GPUs."""
    import ray

    ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            live = [node for node in ray.nodes() if node["Alive"]]
            if len(live) == nodes and sum(node["Resources"].get("GPU", 0) for node in live) >= gpus:
                return
            time.sleep(2)
        raise TimeoutError("Ray workers did not register the expected GPU resources")
    finally:
        ray.shutdown()


def write_runtime_env(path):
    """Write an allowlisted environment privately; never put credentials in argv."""
    keys = {"PYTHONPATH", "RL360_ROOT", "RL360_HF_CHECKPOINT", "RL360_TORCH_CHECKPOINT", "RL360_TASKS_DIR",
            "RL360_MODEL_ARGS_FILE", "RL360_OUTPUT_DIR", "RL360_CONTAINER_IMAGE", "RL360_CONTAINER_MOUNTS",
            "HARBOR_TASKS_DIR", "HARBOR_ENV_TYPE", "HARBOR_DELETE_CONTAINERS", "DAYTONA_API_KEY", "SHARED_DIR",
            "TRAJECTORY_OUTPUT_DIR", "RAY_ADDRESS", "RAY_USAGE_STATS_ENABLED", "AGENT_SERVER_URL", "GATEWAY_URL", "AGENT_MODEL_NAME", "AGENT_TOOL_FORMAT",
            "AGENT_LLM_TIMEOUT_SECS", "MILES_ROUTER_TIMEOUT_SECS", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
            "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF", "WANDB_API_KEY", "WANDB_ENTITY",
            "WANDB_PROJECT", "WANDB_GROUP", "WANDB_DIR", "WANDB_MODE", "WANDB_DISABLE_CODE", "WANDB_INIT_TIMEOUT"}
    env = {key: value for key, value in os.environ.items()
           if key in keys or key.startswith(("NCCL_", "TORCH_NCCL_", "SGLANG_", "MC_"))}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"env_vars": env}, stream)


def write_docker_env(path):
    """Write the same allowlisted environment in Docker env-file format."""
    keys = {"PYTHONPATH", "RL360_ROOT", "RL360_HF_CHECKPOINT", "RL360_TORCH_CHECKPOINT", "RL360_TASKS_DIR",
            "RL360_MODEL_ARGS_FILE", "RL360_OUTPUT_DIR", "RL360_CONTAINER_IMAGE", "RL360_CONTAINER_MOUNTS",
            "HARBOR_TASKS_DIR", "HARBOR_ENV_TYPE", "HARBOR_DELETE_CONTAINERS", "DAYTONA_API_KEY", "SHARED_DIR",
            "TRAJECTORY_OUTPUT_DIR", "RAY_ADDRESS", "RAY_USAGE_STATS_ENABLED", "AGENT_SERVER_URL", "GATEWAY_URL", "AGENT_MODEL_NAME", "AGENT_TOOL_FORMAT",
            "AGENT_LLM_TIMEOUT_SECS", "MILES_ROUTER_TIMEOUT_SECS", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
            "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF", "WANDB_API_KEY", "WANDB_ENTITY",
            "WANDB_PROJECT", "WANDB_GROUP", "WANDB_DIR", "WANDB_MODE", "WANDB_DISABLE_CODE", "WANDB_INIT_TIMEOUT"}
    env = {key: value for key, value in os.environ.items()
           if key in keys or key.startswith(("NCCL_", "TORCH_NCCL_", "SGLANG_", "MC_"))}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        for key, value in sorted(env.items()):
            stream.write(f"{key}={value}\n")


def submit_training(address, runtime_env_path, trainer_args):
    """Submit the trainer, stream its logs, and preserve failure status."""
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(address)
    runtime_env = json.loads(Path(runtime_env_path).read_text())
    # Ray starts the driver through a POSIX shell. Quote JSON and paths once.
    job_id = client.submit_job(
        entrypoint=shlex.join(["python3", "-m", "train_async", *trainer_args]),
        runtime_env=runtime_env,
    )
    print(f"Ray job: {job_id}", flush=True)

    async def follow():
        async for chunk in client.tail_job_logs(job_id):
            print(chunk, end="", flush=True)

    asyncio.run(follow())
    status = client.get_job_status(job_id)
    print(f"\nRay job status: {status}", flush=True)
    return 0 if status == JobStatus.SUCCEEDED else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-inputs", help="Check paths, tasks and model arguments")
    validate.add_argument("--print-model-args", action="store_true", help="Emit one argument per line for Bash")
    container = commands.add_parser("check-container", help="Check imports, mounts and GPUs inside a container")
    container.add_argument("--gpus-per-node", type=int, required=True)
    http = commands.add_parser("wait-http", help="Wait for an HTTP readiness endpoint")
    http.add_argument("url")
    http.add_argument("--timeout", type=int, default=600)
    ray = commands.add_parser("wait-ray", help="Wait for the allocated Ray resources")
    ray.add_argument("--nodes", type=int, required=True)
    ray.add_argument("--gpus", type=int, required=True)
    ray.add_argument("--timeout", type=int, default=600)
    runtime = commands.add_parser("write-runtime-env", help="Write a private Ray environment file")
    runtime.add_argument("path")
    docker_env = commands.add_parser("write-docker-env", help="Write a private Docker environment file")
    docker_env.add_argument("path")
    submit = commands.add_parser("submit-training", help="Run the trainer and follow its logs")
    submit.add_argument("--address", required=True)
    submit.add_argument("--runtime-env", required=True)
    submit.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == "validate-inputs":
            model_args = validate_inputs()
            print("\n".join(model_args) if args.print_model_args else "Validated 32 tasks, checkpoint paths and model arguments.")
        elif args.command == "check-container":
            check_container(args.gpus_per_node)
        elif args.command == "wait-http":
            wait_http(args.url, args.timeout)
        elif args.command == "wait-ray":
            wait_ray(args.nodes, args.gpus, args.timeout)
        elif args.command == "write-runtime-env":
            write_runtime_env(args.path)
        elif args.command == "write-docker-env":
            write_docker_env(args.path)
        elif args.command == "submit-training":
            trainer_args = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
            require(trainer_args, "Pass trainer arguments after --")
            return submit_training(args.address, args.runtime_env, trainer_args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
