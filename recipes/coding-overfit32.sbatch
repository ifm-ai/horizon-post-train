#!/usr/bin/env bash
#SBATCH --job-name=coding-overfit32
#SBATCH --nodes=33
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%j.log

# Adapted from a mixed-domain, 32-task overfit experiment. See recipes/README.md.
# Supply site-specific account/partition/QoS with sbatch, not in this file.
# The 33-node default uses one service node, eight actor nodes, and 24 rollout
# nodes. Model architecture, tool parsers, checkpoints and tasks are user inputs.
set +x
set -euo pipefail
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
        printf '%s\n' 'Usage: bash recipes/coding-overfit32.sbatch --dry-run' \
            '       sbatch [site/resource options] recipes/coding-overfit32.sbatch' \
            'Configuration and prerequisites: recipes/README.md'
        exit 0 ;;
esac
[[ $# == 0 ]] || die 'Unexpected arguments; use environment variables documented in recipes/README.md.'

export RL360_ROOT="${RL360_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
for name in RL360_CONTAINER_IMAGE RL360_CONTAINER_MOUNTS RL360_HF_CHECKPOINT \
    RL360_TORCH_CHECKPOINT RL360_TASKS_DIR RL360_MODEL_ARGS_FILE HARBOR_ENV_TYPE; do
    [[ -n "${!name:-}" ]] || die "Set $name (see recipes/README.md)."
    export "$name"
done
[[ -f "$RL360_ROOT/components/miles/train_async.py" ]] || die 'RL360_ROOT must point to this source snapshot.'
export RL360_OUTPUT_DIR="${RL360_OUTPUT_DIR:-$RL360_ROOT/outputs}"

ACTOR_NODES="${RL360_ACTOR_NODES:-8}"
GPUS_PER_NODE="${RL360_GPUS_PER_NODE:-8}"
ROLLOUT_GPUS="${RL360_ROLLOUT_GPUS:-192}"
ENGINE_GPUS="${RL360_ENGINE_GPUS:-2}"
TP="${RL360_TP:-2}"
CP="${RL360_CP:-4}"
NUM_ROLLOUT="${RL360_NUM_ROLLOUT:-100}"
HARBOR_WORKERS="${RL360_HARBOR_WORKERS:-256}"
HARBOR_CONCURRENCY="${RL360_HARBOR_CONCURRENCY:-40}"
SESSION_SERVERS="${RL360_SESSION_SERVERS:-256}"
CONTEXT_LENGTH="${RL360_CONTEXT_LENGTH:-131072}"
MAX_RESPONSE_LEN="${RL360_MAX_RESPONSE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${RL360_MAX_TOKENS_PER_GPU:-32768}"
STARTUP_TIMEOUT="${RL360_STARTUP_TIMEOUT:-600}"
RAY_PORT="${RL360_RAY_PORT:-6379}"
DASHBOARD_PORT="${RL360_DASHBOARD_PORT:-8265}"
GATEWAY_PORT="${RL360_GATEWAY_PORT:-8080}"
HARBOR_PORT="${RL360_HARBOR_PORT:-11000}"
for name in ACTOR_NODES GPUS_PER_NODE ROLLOUT_GPUS ENGINE_GPUS TP CP NUM_ROLLOUT \
    HARBOR_WORKERS HARBOR_CONCURRENCY SESSION_SERVERS CONTEXT_LENGTH MAX_RESPONSE_LEN \
    MAX_TOKENS_PER_GPU STARTUP_TIMEOUT RAY_PORT DASHBOARD_PORT GATEWAY_PORT HARBOR_PORT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done
(( ROLLOUT_GPUS % GPUS_PER_NODE == 0 )) || die 'ROLLOUT_GPUS must fill whole rollout nodes.'
(( GPUS_PER_NODE % ENGINE_GPUS == 0 )) || die 'ENGINE_GPUS must divide GPUS_PER_NODE.'
(( ACTOR_NODES * GPUS_PER_NODE % (TP * CP) == 0 )) || die 'Actor GPU count must be divisible by TP * CP.'
EXPECTED_NODES=$((1 + ACTOR_NODES + ROLLOUT_GPUS / GPUS_PER_NODE))

# Input validation is shared with the standalone preflight command.
RUNTIME_HELPER="$RL360_ROOT/recipes/overfit_runtime.py"
MODEL_ARGS_TEXT=$(python3 "$RUNTIME_HELPER" validate-inputs --print-model-args)
mapfile -t MODEL_ARGS <<< "$MODEL_ARGS_TEXT"

case "$HARBOR_ENV_TYPE" in
    docker) ;; # Docker Engine and NVIDIA Container Toolkit must be available on every node.
    daytona) [[ -n "${DAYTONA_API_KEY:-}" || $DRY_RUN == 1 ]] || die 'Set DAYTONA_API_KEY.' ;;
    *) die 'This launcher supports HARBOR_ENV_TYPE=docker or daytona.' ;;
esac

HEAD_IP="${RL360_HEAD_IP:-127.0.0.1}"
RUN_DIR="$RL360_OUTPUT_DIR/${SLURM_JOB_ID:-dry-run}"
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    : "${WANDB_ENTITY:?Set WANDB_ENTITY when WANDB_PROJECT is configured}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_GROUP="${WANDB_GROUP:-coding-overfit32-${SLURM_JOB_ID:-dry-run}}"
    if [[ "$WANDB_MODE" == online && $DRY_RUN == 0 ]]; then
        [[ -n "${WANDB_API_KEY:-}" ]] || die 'Export WANDB_API_KEY for online logging across nodes.'
    fi
    WANDB_ARGS=(--use-wandb --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT"
        --wandb-group "$WANDB_GROUP"
        --wandb-mode "$WANDB_MODE" --wandb-dir "$RUN_DIR/wandb")
else
    export WANDB_MODE=disabled
fi

if (( ! DRY_RUN )); then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die 'Submit with sbatch, or use --dry-run.'
    NODE_TEXT=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t NODES <<< "$NODE_TEXT"
    [[ ${#NODES[@]} == "$EXPECTED_NODES" ]] || die "Request $EXPECTED_NODES nodes for the configured GPU split."
    HEAD_NODE="${NODES[0]}"
    HEAD_IP="${RL360_HEAD_IP:-$(getent ahostsv4 "$HEAD_NODE" | awk 'NR == 1 {print $1}')}"
    [[ -n "$HEAD_IP" ]] || die 'Cannot resolve head node; set RL360_HEAD_IP.'
fi

TRAIN_ARGS=(
    --actor-num-nodes "$ACTOR_NODES" --actor-num-gpus-per-node "$GPUS_PER_NODE"
    --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine "$ENGINE_GPUS"
    --num-rollout "$NUM_ROLLOUT" --update-weights-interval 1
    --rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256
    --use-dynamic-global-batch-size --rollout-max-response-len "$MAX_RESPONSE_LEN"
    --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_valid_reward_nonzero_std
    --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_infra_failures
    --tensor-model-parallel-size "$TP" --context-parallel-size "$CP"
    --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --hf-checkpoint "$RL360_HF_CHECKPOINT" --ref-load "$RL360_TORCH_CHECKPOINT"
    --save "$RUN_DIR/checkpoints" --save-interval 10
    --optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
    --advantage-estimator grpo --disable-grpo-std-normalization --use-tis
    --use-kl-loss --kl-loss-coef 0 --kl-loss-type low_var_kl
    --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28
    --sequence-parallel --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --attention-backend flash
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --attention-dropout 0 --hidden-dropout 0 --keep-old-actor
    --miles-router-timeout 600 --sglang-mem-fraction-static 0.75 --sglang-page-size 64
    --sglang-max-running-requests 128 --sglang-chunked-prefill-size 4096
    --sglang-watchdog-timeout 3600 --pause-generation-mode retract
    --sglang-dp-size 1 --sglang-served-model-name model --sglang-context-length "$CONTEXT_LENGTH"
    --sglang-router-ip "$HEAD_IP" --sglang-router-port "$GATEWAY_PORT"
    --sglang-enable-hierarchical-cache --sglang-hicache-io-backend kernel --sglang-hicache-ratio 1
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path agent360.harbor.miles.swe_agent_function.run
    --custom-agent-config-json '{"agent_overrides":{"repetitive_tool_call_failure":{"enabled":true,"threshold":3},"format_error_reward_penalty":{"value":0.3,"apply_to":"successful"}}}'
    --custom-rm-path agent360.harbor.miles.generate.reward_func
    --rollout-function-path agent360.harbor.miles.generate.RolloutFn
    --rollout-all-samples-process-path agent360.harbor.miles.generate.log_all_samples
    --prompt-data "$RL360_TASKS_DIR/harbor_records.jsonl" --input-key prompt --metadata-key metadata
    --rollout-shuffle --rollout-temperature 1 --rollout-top-p 1
    --use-dynamic-batch-size --balance-data --balance-by-flops
    --disable-oversampling --tail-cancel-groups 0 --rolling-start-size 512 --rolling-start-interval 30
    --chat-template-path autofix --use-session-server --session-server-count "$SESSION_SERVERS"
    --tito-allowed-append-roles tool user system
    "${MODEL_ARGS[@]}" "${WANDB_ARGS[@]}"
)
if (( DRY_RUN )); then
    printf 'Inputs validated. Allocation: %s nodes; %s actor GPUs, %s rollout GPUs, one service node.\n' \
        "$EXPECTED_NODES" "$((ACTOR_NODES * GPUS_PER_NODE))" "$ROLLOUT_GPUS"
    printf 'Dry-run trainer command (head IP is a placeholder):\n'
    printf '%q ' python3 -m train_async "${TRAIN_ARGS[@]}"
    printf '\nNo job submitted. Container dependencies, GPU memory and sandbox connectivity are not checked.\n'
    exit 0
fi

mkdir -p "$RL360_OUTPUT_DIR"
mkdir "$RUN_DIR"  # Never overwrite an existing run.
mkdir "$RUN_DIR/wandb"
export PYTHONPATH="$RL360_ROOT/src:$RL360_ROOT/components/miles:$RL360_ROOT/components/megatron-lm:$RL360_ROOT/components/sglang/python:$RL360_ROOT/components/harbor/src${PYTHONPATH:+:$PYTHONPATH}"
export HARBOR_TASKS_DIR="$RL360_TASKS_DIR" HARBOR_DELETE_CONTAINERS=true
export SHARED_DIR="$RUN_DIR" TRAJECTORY_OUTPUT_DIR="$RUN_DIR/trajectories"
export AGENT_SERVER_URL="http://$HEAD_IP:$HARBOR_PORT" GATEWAY_URL="http://$HEAD_IP:$GATEWAY_PORT"
export AGENT_MODEL_NAME=model AGENT_LLM_TIMEOUT_SECS=600 MILES_ROUTER_TIMEOUT_SECS=600 AGENT_TOOL_FORMAT=
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 CUDA_DEVICE_MAX_CONNECTIONS=1 SGLANG_FORWARD_UNKNOWN_TOOLS=1
export SGLANG_LOAD_WEIGHT_TIMEOUT_SECS="${SGLANG_LOAD_WEIGHT_TIMEOUT_SECS:-600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="$RUN_DIR/wandb" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT" RAY_USAGE_STATS_ENABLED=0
export SLURM_EXPORT_ENV=ALL
RUNTIME_ENV_FILE="$RUN_DIR/runtime-env.json"
DOCKER_ENV_FILE="$RUN_DIR/docker.env"
DOCKER_MOUNTS=()
IFS=',' read -ra MOUNT_SPECS <<< "$RL360_CONTAINER_MOUNTS"
for mount in "${MOUNT_SPECS[@]}"; do
    [[ -n "$mount" ]] || continue
    DOCKER_MOUNTS+=(--volume "$mount")
done
SRUN=(srun --overlap --nodes=1 --ntasks=1 --export=ALL)
HEAD_CONTAINER=(docker run --rm --network host --ipc host --env-file "$DOCKER_ENV_FILE" "${DOCKER_MOUNTS[@]}" --workdir "$RL360_ROOT" "$RL360_CONTAINER_IMAGE")
GPU_CONTAINER=(docker run --rm --network host --ipc host --gpus "$GPUS_PER_NODE" --env-file "$DOCKER_ENV_FILE" "${DOCKER_MOUNTS[@]}" --workdir "$RL360_ROOT" "$RL360_CONTAINER_IMAGE")
PIDS=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    # Only signal the Slurm steps started by this launcher. Slurm owns their
    # process trees; do not kill unrelated Python/Ray processes on the machine.
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    if (( ${#PIDS[@]} )); then
        sleep 2
        for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    fi
    wait 2>/dev/null || true
    rm -f -- "$RUNTIME_ENV_FILE" "$DOCKER_ENV_FILE"  # May contain user-supplied W&B credentials.
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The helper writes private files used by Ray and Docker; neither contains credentials in argv.
python3 "$RUNTIME_HELPER" write-runtime-env "$RUNTIME_ENV_FILE"
python3 "$RUNTIME_HELPER" write-docker-env "$DOCKER_ENV_FILE"

# Check the mounted sources and compiled dependencies on every node first.
srun --nodes="$EXPECTED_NODES" --ntasks="$EXPECTED_NODES" --export=ALL \
    "${GPU_CONTAINER[@]}" python3 "$RUNTIME_HELPER" check-container \
    --gpus-per-node "$GPUS_PER_NODE" > "$RUN_DIR/preflight.log" 2>&1

start_step() {
    local node=$1 log=$2 mode=$3
    shift 3
    local -a container=("${HEAD_CONTAINER[@]}")
    [[ "$mode" == gpu ]] && container=("${GPU_CONTAINER[@]}")
    "${SRUN[@]}" --nodelist="$node" "${container[@]}" "$@" > "$RUN_DIR/$log" 2>&1 &
    PIDS+=("$!")
}
wait_http() {
    python3 "$RUNTIME_HELPER" wait-http "$1" --timeout "$STARTUP_TIMEOUT"
}

start_step "$HEAD_NODE" ray-head.log cpu ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
    --temp-dir="/tmp/rl360-ray-$SLURM_JOB_ID" --num-gpus=0 --disable-usage-stats \
    --dashboard-host="$HEAD_IP" --dashboard-port="$DASHBOARD_PORT" --block
wait_http "http://$HEAD_IP:$DASHBOARD_PORT/api/version"
for node in "${NODES[@]:1}"; do
    start_step "$node" "ray-$node.log" gpu ray start --address="$RAY_ADDRESS" \
        --num-gpus="$GPUS_PER_NODE" --disable-usage-stats --block
done
"${SRUN[@]}" --nodelist="$HEAD_NODE" "${HEAD_CONTAINER[@]}" python3 "$RUNTIME_HELPER" wait-ray \
    --nodes "$EXPECTED_NODES" --gpus "$((ACTOR_NODES * GPUS_PER_NODE + ROLLOUT_GPUS))" \
    --timeout "$STARTUP_TIMEOUT"

start_step "$HEAD_NODE" gateway.log cpu python3 -m smg.launch_router --host "$HEAD_IP" --port "$GATEWAY_PORT" \
    --policy round_robin --assignment-mode random --disable-circuit-breaker --disable-retries \
    --tokenizer-path "$RL360_HF_CHECKPOINT" --log-level info
wait_http "$GATEWAY_URL/health"
start_step "$HEAD_NODE" harbor.log cpu python3 -m agent360.harbor.miles.harbor_server \
    --port "$HARBOR_PORT" --workers "$HARBOR_WORKERS" --max-concurrent "$HARBOR_CONCURRENCY" --log-level INFO
wait_http "$AGENT_SERVER_URL/health"

printf 'Starting training. Logs and checkpoints: %s\n' "$RUN_DIR"
"${SRUN[@]}" --nodelist="$HEAD_NODE" "${HEAD_CONTAINER[@]}" python3 "$RUNTIME_HELPER" submit-training \
    --address "http://$HEAD_IP:$DASHBOARD_PORT" --runtime-env "$RUNTIME_ENV_FILE" \
    -- "${TRAIN_ARGS[@]}" > "$RUN_DIR/trainer.log" 2>&1
