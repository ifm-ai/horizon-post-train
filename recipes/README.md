# K2 Horizon 7B: 32-task coding overfit

View the reference run and its pinned charts in the [public W&B workspace](https://wandb.ai/mbzuai-llm/rl360-public).

[`coding-overfit32.sbatch`](coding-overfit32.sbatch) runs the coding overfit recipe used with a K2 Horizon 7B training checkpoint. It starts Ray, SMG, and Harbor, then launches the Miles asynchronous trainer. The defaults are:

| Setting | Default |
| --- | --- |
| Task pool / samples per task | 32 / 8 |
| Planned rollout iterations | 100 |
| Global batch size | Up to 256, adjusted after filtering |
| Optimizer | Adam, constant `5e-6`, betas `0.9 / 0.98`, weight decay `0.1` |
| Objective | GRPO, standard-deviation normalization disabled, importance sampling enabled |
| PPO clipping / KL coefficient | `0.2 / 0.28` / `0` |
| Context / maximum response tokens | 131072 / 8192 |
| Actor GPUs / rollout GPUs | 64 / 192 |
| Actor tensor / context parallelism | 2 / 4 |
| GPUs per inference engine | 2 |
| Checkpoint interval | Every 10 rollout iterations |
| Format penalty | `0.3`, applied to otherwise successful attempts |
| Repeated tool-failure threshold | 3 |

Groups need at least two valid, differing rewards. Infrastructure failures are masked from the loss; policy failures and token truncations remain in the recipe. The learning curves therefore depend on task selection and verifier behavior.

The `.sbatch` file contains the configuration, trainer arguments, and service launch sequence. [`overfit_runtime.py`](overfit_runtime.py) handles input checks, service readiness, and Ray job submission. Its commands are available through `python recipes/overfit_runtime.py --help`; validation does not require importing the GPU stack.

## Prerequisites

- Slurm with Docker Engine, the NVIDIA Container Toolkit, and an exclusive allocation. Load the image on every allocated node before submitting the job.
- A container with Python 3.12, Ray, compatible CUDA/PyTorch/Megatron/SGLang dependencies, and the compiled `smg.smg_rs` extension built for the bundled SMG. The [main README](../README.md#run-the-coding-recipe) provides a public container build, checkpoint conversion commands, and a launch walkthrough. That adapted build still needs end-to-end validation on the target cluster.
- Shared absolute paths, visible at the same locations on every node and in each container, for this repository, checkpoints, task data, and outputs.
- An HF checkpoint containing model configuration, tokenizer and weights, plus the matching converted Megatron checkpoint. `--ref-load` initializes the reference policy and supplies the initial actor weights through Miles' normal checkpoint fallback. This launcher starts a new run; it does not resume one.
- A configured sandbox backend with verified tasks. Run each task's oracle and verifier before using it for learning. Sandbox provisioning is separate from the Slurm GPU allocation.

The launcher's readiness checks cover imports, GPU allocation and service startup. They do not establish checkpoint compatibility, sufficient GPU/NUMA memory, container disk space, or verifier correctness. The included [`memory_preflight.py`](../src/agent360/harbor/miles/memory_preflight.py) and [`storage_preflight.py`](../src/agent360/harbor/miles/storage_preflight.py) can be run with thresholds suited to your hardware; storage checks require an explicit `--path` for the container extraction filesystem.

## Model arguments

The [main README](../README.md#run-the-coding-recipe) uses the public [K2 Horizon 7B](https://huggingface.co/IFM/K2-Horizon-7B) checkpoint with [`k2-horizon-7b.json`](k2-horizon-7b.json) and `RL360_CONTEXT_LENGTH=131072`. The JSON supplies the model's Megatron architecture, tokenizer, and parser settings. It includes grouped RMSNorm with four groups, a vocabulary of 250624 tokens, and a RoPE base of 10000000.

Set `RL360_MODEL_ARGS_FILE` to the JSON file for your model. Each entry is a command-line argument; numbers must be strings too. To add another model, copy the closest configuration and update its dimensions, query groups, normalization, positional embeddings, vocabulary, weight tying, and parsers to match the checkpoint.

Tokenizer types are defined in [`tito_tokenizer.py`](../components/miles/miles/utils/chat_template_utils/tito_tokenizer.py), and tool parsers are in [`function_call/`](../components/sglang/python/sglang/srt/function_call/). The launcher uses `--chat-template-path autofix`; add a different `--chat-template-path` to the model JSON if needed. Keep allocation and training settings in the launcher environment variables.

### K2 Horizon checkpoint format

The public checkpoint uses a newer Transformers configuration API and calls its fast tokenizer `TokenizersBackend`. [`prepare_k2_horizon.py`](prepare_k2_horizon.py) adapts these metadata files for the version used by the bundled training stack. It checks the 7B architecture, points `AutoConfig` to a compatible configuration class, and selects `PreTrainedTokenizerFast` with the original tokenizer vocabulary and chat template. The prepared directory is for RL360's native Megatron/SGLang path; use the original download with the model card's Transformers examples.

The command requires a new output directory and leaves the source download intact. It hard-links unchanged files when both directories are on the same filesystem, or copies them across filesystems. Keep checkpoint files read-only after preparation. The prepared configuration retains the public `k2_horizon` model type and `K2HorizonForCausalLM` architecture. The bundled runtime calls the matching tokenizer and parsers `k2v3` and `k2_v3`.

### Qwen3-8B

To use [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), build the same container and replace the K2 checkpoint download and conversion commands with:

```bash
docker run --rm -v /shared:/shared rl360:local \
  hf download Qwen/Qwen3-8B --local-dir /shared/models/qwen3-8b-hf

docker run --rm --gpus all --ipc=host -v /shared:/shared rl360:local \
  torchrun --standalone --nproc-per-node=8 components/miles/tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /shared/models/qwen3-8b-hf \
  --save /shared/models/qwen3-8b-megatron \
  --swiglu --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --use-rotary-position-embeddings --rotary-base 1000000 \
  --disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6 \
  --vocab-size 151936 --kv-channels 128 --qk-layernorm \
  --untie-embeddings-and-output-weights
```

Set the following before running the dry run or submitting the job:

```bash
export RL360_HF_CHECKPOINT=/shared/models/qwen3-8b-hf
export RL360_TORCH_CHECKPOINT=/shared/models/qwen3-8b-megatron
export RL360_MODEL_ARGS_FILE="$RL360_ROOT/recipes/qwen3-8b.json"
export RL360_CONTEXT_LENGTH=32768
```

[`qwen3-8b.json`](qwen3-8b.json) selects Qwen's architecture and parsers. The 32K context setting stays within its [40960-token model configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json). Task preparation and job submission are the same as in the K2 example.

## Tasks

Provide a directory containing exactly 32 unique records and their Harbor tasks:

```text
tasks/
  harbor_records.jsonl
  task-001/
    instruction.md
    task.toml
    environment/       # Environment build/configuration for the chosen backend
    tests/test.sh      # Writes verifier reward using Harbor's task contract
    solution/          # Oracle solution, when available
  ...
```

Each JSONL row uses this structure:

```json
{"prompt":"Solve the task.","metadata":{"instance_id":"task-001","tags":["coding"]}}
```

`prompt` can also be a chat-message list. `instance_id` names the task directory; the first tag supplies the domain label used in reward metrics. Symlinked task directories are supported when their targets are also mounted in the container. Task-specific external datasets and images must be accessible to the sandbox. No task contents, selection manifests, model weights or historical trajectories are bundled here.

The original overfit protocol selected tasks using 16 attempts per task from the initial policy, retaining tasks with a measured pass rate in the 20–40% range and no infrastructure failures. That selection affects the result. This launcher assumes tasks are already selected; it does not run calibration or submit a follow-up job. The calibration implementation is available in [`calibration.py`](../src/agent360/harbor/miles/calibration.py).

## Configure and launch

Run from the repository root. Replace these example paths with your own shared paths and include every input/output path in the container mount specification:

```bash
export RL360_ROOT="$PWD"
export RL360_CONTAINER_IMAGE=rl360:local
export RL360_CONTAINER_MOUNTS=/shared:/shared
export RL360_HF_CHECKPOINT=/shared/models/k2-horizon-7b-hf
export RL360_TORCH_CHECKPOINT=/shared/models/k2-horizon-7b-megatron
export RL360_TASKS_DIR=/shared/data/overfit32
export RL360_MODEL_ARGS_FILE="$RL360_ROOT/recipes/k2-horizon-7b.json"
export RL360_OUTPUT_DIR=/shared/outputs/rl360
export RL360_CONTEXT_LENGTH=131072
export HARBOR_ENV_TYPE=daytona
```

`RL360_ROOT` must also be under a mounted path. Set sandbox credentials through the environment before submission; keep credentials and machine-specific input files outside the repository.

| Backend | Required setup |
| --- | --- |
| `daytona` | `DAYTONA_API_KEY` and suitable sandbox capacity |
| `docker` | Docker CLI/Compose and a reachable daemon inside the training container; configure `DOCKER_HOST` or mount the daemon socket as appropriate |

W&B logging is opt-in: set `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_API_KEY`. Optional `WANDB_GROUP` sets the run name/group. `WANDB_MODE=offline` permits local logging without an API key. Without a project, W&B logging is disabled. Credentials are passed through the runtime environment, not trainer arguments. The bundled W&B integration uses an explicit configuration allowlist for both primary and worker processes and disables automatic source/machine metadata and console capture. Task tags still become metric names; use public labels when sharing runs. Ray and the other services should be reachable only within your trusted cluster.

```bash
# Checks input files and prints settings; does not start services or submit jobs.
bash recipes/coding-overfit32.sbatch --dry-run

# Add the account, partition and QoS required by your site.
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION recipes/coding-overfit32.sbatch
```

The defaults allocate **33 nodes with eight GPUs each**. The service node's GPUs are unused. To change the allocation, set `RL360_ACTOR_NODES`, `RL360_GPUS_PER_NODE`, `RL360_ROLLOUT_GPUS`, and `RL360_ENGINE_GPUS`, and pass matching `sbatch --nodes` and `--gpus-per-node` options. Required nodes are `1 + actor_nodes + rollout_gpus / gpus_per_node`.

Other overrides: `RL360_TP`, `RL360_CP`, `RL360_NUM_ROLLOUT`, `RL360_CONTEXT_LENGTH`, `RL360_MAX_RESPONSE_LEN`, `RL360_MAX_TOKENS_PER_GPU`, `RL360_HARBOR_WORKERS`, `RL360_HARBOR_CONCURRENCY` (per worker), `RL360_SESSION_SERVERS`, and `RL360_STARTUP_TIMEOUT` (seconds). Head address and ports can be set with `RL360_HEAD_IP`, `RL360_RAY_PORT`, `RL360_DASHBOARD_PORT`, `RL360_GATEWAY_PORT`, and `RL360_HARBOR_PORT`. Export your site's `NCCL_*`/`TORCH_NCCL_*` settings when needed; no network-interface names or fabric-specific settings are assumed.

## Outputs and interpretation

Logs, trajectories and checkpoints go under `RL360_OUTPUT_DIR/<job-id>/`. The script stops its own Slurm service steps when training exits. Logs and task traces are local run artifacts; inspect them before sharing.

For learning progress, plot `reward/unfiltered_mean` against `rollout/step`. This includes groups discarded by dynamic filtering; `reward/kept_mean` describes the selected training batch and can move differently as easy groups disappear. Format penalties mean a reward mean is not an exact verifier success rate. Per-domain means and counts are available under `reward/domain/` using the task tags. `rollout/samples_masked` and `rollout/samples_trained` help interpret the effective batch.

This adapter has local syntax, input-validation and dependency checks. Its full Slurm/container/GPU execution still needs validation in your deployment.
