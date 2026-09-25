from __future__ import annotations

import logging
import random
import socket
from argparse import Namespace
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.ci_utils import assert_rollout_engine_weight_versions
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint
from .initialize import init, is_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state
from .replay_utils import get_register_replay_list_func
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import math


def validate_rollout_for_grpo_training_step(
    args,
    rollout_data,
    *,
    rollout_id=None,
    where="train_actor.begin",
    logger=None,
    require_log_probs=False,
):
    """
    Local-only rollout validator for MegatronTrainRayActor.

    No collectives. No mutation. Safe to run independently on every rank.
    Logs useful diagnostics before raising so NCCL-desync root cause is visible
    in the first failing rank's log.
    """
    import socket
    import traceback

    import torch
    import torch.distributed as dist

    errors = []
    warnings = []

    def _safe_get_parallel_state():
        try:
            # actor.py already imports get_parallel_state from
            # miles.backends.training_utils.parallel
            return get_parallel_state()
        except Exception as e:
            warnings.append(f"get_parallel_state() failed: {type(e).__name__}: {e}")
            return None

    ps = _safe_get_parallel_state()

    def _maybe_attr(obj, *path, default=None):
        cur = obj
        for p in path:
            try:
                cur = getattr(cur, p)
            except Exception:
                return default
        return cur

    def _rank_info():
        parts = []
        try:
            parts.append(f"host={socket.gethostname()}")
        except Exception:
            pass

        try:
            if dist.is_available() and dist.is_initialized():
                parts.append(f"global_rank={dist.get_rank()}/{dist.get_world_size()}")
            else:
                parts.append("global_rank=dist_not_initialized")
        except Exception as e:
            parts.append(f"global_rank=unavailable:{type(e).__name__}")

        if ps is not None:
            for name in ("dp", "intra_dp", "cp", "tp", "pp", "ep"):
                group_state = getattr(ps, name, None)
                if group_state is not None:
                    r = getattr(group_state, "rank", "?")
                    s = getattr(group_state, "size", "?")
                    parts.append(f"{name}_rank={r}/{s}")

        # Fallbacks from args. These are less authoritative than ps.
        for arg_name in (
            "data_parallel_size",
            "context_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
        ):
            if hasattr(args, arg_name):
                parts.append(f"args.{arg_name}={getattr(args, arg_name)}")

        return " ".join(parts)

    rank_info = _rank_info()

    def _log(level, msg):
        prefix = f"ROLLOUT_VALIDATE {where}"
        if rollout_id is not None:
            prefix += f" rollout_id={rollout_id}"
        prefix += f" {rank_info} :: {msg}"

        if logger is not None:
            getattr(logger, level)(prefix)
        else:
            print(prefix, flush=True)

    def _add_error(msg):
        errors.append(msg)

    def _add_warning(msg):
        warnings.append(msg)

    def _is_seq(x):
        return isinstance(x, (list, tuple))

    def _present(key):
        return key in rollout_data and rollout_data[key] is not None

    def _numel(x):
        if torch.is_tensor(x):
            return int(x.numel())
        if isinstance(x, (list, tuple)):
            return len(x)
        return None

    def _ndim(x):
        if torch.is_tensor(x):
            return int(x.ndim)
        if isinstance(x, (list, tuple)):
            return 1
        return None

    def _finite_tensor_or_list(x):
        try:
            if torch.is_tensor(x):
                return bool(torch.isfinite(x.float()).all().item())
            if isinstance(x, (list, tuple)):
                return all(math.isfinite(float(v)) for v in x)
            if isinstance(x, (float, int)):
                return math.isfinite(float(x))
            return False
        except Exception:
            return False

    def _sum_float(x):
        if torch.is_tensor(x):
            return float(x.float().sum().item())
        if isinstance(x, (list, tuple)):
            return float(sum(float(v) for v in x))
        return float(x)

    def _all_zero(x):
        if torch.is_tensor(x):
            return bool(torch.all(x == 0).item())
        if isinstance(x, (list, tuple)):
            return all(float(v) == 0 for v in x)
        return float(x) == 0

    def _summarize_vector_list(key, limit=3):
        if not _present(key):
            return f"{key}=MISSING"
        xs = rollout_data[key]
        if not _is_seq(xs):
            return f"{key}=BAD_TYPE({type(xs).__name__})"

        shapes = []
        dtypes = []
        devices = []
        samples = min(len(xs), limit)
        for i in range(samples):
            x = xs[i]
            if torch.is_tensor(x):
                shapes.append(tuple(x.shape))
                dtypes.append(str(x.dtype))
                devices.append(str(x.device))
            elif isinstance(x, (list, tuple)):
                shapes.append((len(x),))
                dtypes.append(type(x[0]).__name__ if len(x) else "empty")
                devices.append("python")
            else:
                shapes.append(type(x).__name__)
                dtypes.append(type(x).__name__)
                devices.append("python")
        return f"{key}: len={len(xs)} first_shapes={shapes} " f"first_dtypes={dtypes} first_devices={devices}"

    def _basic_batch_summary():
        keys = sorted(list(rollout_data.keys()))
        key_summary = "keys=" + ",".join(keys)

        lines = [key_summary]

        for key in (
            "rewards",
            "response_lengths",
            "total_lengths",
            "removed_by_filter",
            "loss_masks",
            "tokens",
            "input_ids",
            "max_seq_lens",
            "log_probs",
            "rollout_log_probs",
            "ref_log_probs",
            "values",
            "advantages",
            "returns",
        ):
            lines.append(_summarize_vector_list(key))

        # Numeric aggregate summary.
        try:
            if _present("removed_by_filter") and _is_seq(rollout_data["removed_by_filter"]):
                flags = rollout_data["removed_by_filter"]
                lines.append(f"removed_by_filter: count={len(flags)} " f"removed={sum(bool(flag) for flag in flags)}")
        except Exception as e:
            lines.append(f"removed_by_filter aggregate failed: {type(e).__name__}: {e}")

        try:
            if _present("response_lengths") and _is_seq(rollout_data["response_lengths"]):
                rs = [int(x) for x in rollout_data["response_lengths"]]
                lines.append(
                    f"response_lengths: count={len(rs)} sum={sum(rs)} "
                    f"min={min(rs) if rs else None} max={max(rs) if rs else None}"
                )
        except Exception as e:
            lines.append(f"response_lengths aggregate failed: {type(e).__name__}: {e}")

        try:
            if _present("total_lengths") and _is_seq(rollout_data["total_lengths"]):
                ts = [int(x) for x in rollout_data["total_lengths"]]
                lines.append(
                    f"total_lengths: count={len(ts)} sum={sum(ts)} "
                    f"min={min(ts) if ts else None} max={max(ts) if ts else None}"
                )
        except Exception as e:
            lines.append(f"total_lengths aggregate failed: {type(e).__name__}: {e}")

        try:
            if _present("loss_masks") and _is_seq(rollout_data["loss_masks"]):
                ms = [_sum_float(m) for m in rollout_data["loss_masks"]]
                lines.append(
                    f"loss_mask_sums: count={len(ms)} sum={sum(ms):.1f} "
                    f"min={min(ms) if ms else None} max={max(ms) if ms else None}"
                )
        except Exception as e:
            lines.append(f"loss_masks aggregate failed: {type(e).__name__}: {e}")

        try:
            if _present("rewards") and _is_seq(rollout_data["rewards"]):
                rw = [float(x) for x in rollout_data["rewards"]]
                lines.append(
                    f"rewards: count={len(rw)} sum={sum(rw):.6g} "
                    f"min={min(rw) if rw else None} max={max(rw) if rw else None}"
                )
        except Exception as e:
            lines.append(f"rewards aggregate failed: {type(e).__name__}: {e}")

        return " | ".join(lines)

    # ------------------------------------------------------------------
    # Start diagnostics.
    # ------------------------------------------------------------------

    _log(
        "warning",
        "start "
        f"advantage_estimator={getattr(args, 'advantage_estimator', None)} "
        f"normalize_advantages={getattr(args, 'normalize_advantages', None)} "
        f"use_rollout_logprobs={getattr(args, 'use_rollout_logprobs', None)} "
        f"use_critic={getattr(args, 'use_critic', None)} "
        f"qkv_format={getattr(args, 'qkv_format', None)} "
        f"compute_advantages_and_returns={getattr(args, 'compute_advantages_and_returns', None)} "
        f"n_samples_per_prompt={getattr(args, 'n_samples_per_prompt', None)} "
        f"generate_multi_samples={getattr(args, 'generate_multi_samples', None)}",
    )

    # ------------------------------------------------------------------
    # Required fields.
    # ------------------------------------------------------------------

    required = ("rewards", "response_lengths", "total_lengths", "loss_masks")
    for key in required:
        if not _present(key):
            _add_error(f"missing required key {key!r}")
        elif not _is_seq(rollout_data[key]):
            _add_error(f"{key!r} must be list/tuple, got {type(rollout_data[key]).__name__}")

    if errors:
        _log("error", "summary_before_failure " + _basic_batch_summary())
        for w in warnings:
            _log("warning", w)
        for e in errors:
            _log("error", "failure " + e)
        raise ValueError(f"{where}: rollout validation failed with {len(errors)} error(s); see logs above")

    n = len(rollout_data["rewards"])
    if n == 0:
        _add_error("empty rollout batch: len(rewards)=0")

    for key in ("response_lengths", "total_lengths", "loss_masks"):
        got = len(rollout_data[key])
        if got != n:
            _add_error(f"{key!r} length mismatch: got {got}, expected {n}")

    removed_by_filter_flags = [False] * n
    if _present("removed_by_filter"):
        candidate_flags = rollout_data["removed_by_filter"]
        candidate_flags_valid = True
        if not _is_seq(candidate_flags):
            _add_error(f"'removed_by_filter' must be list/tuple, got {type(candidate_flags).__name__}")
            candidate_flags_valid = False
        elif len(candidate_flags) != n:
            _add_error(f"'removed_by_filter' length mismatch: got {len(candidate_flags)}, expected {n}")
            candidate_flags_valid = False
        else:
            for i, removed in enumerate(candidate_flags):
                if not isinstance(removed, bool):
                    _add_error(f"removed_by_filter[{i}] must be bool, got {type(removed).__name__}")
                    candidate_flags_valid = False
            if candidate_flags_valid:
                removed_by_filter_flags = list(candidate_flags)

    token_key = None
    if _present("tokens"):
        token_key = "tokens"
    elif _present("input_ids"):
        token_key = "input_ids"

    if token_key is None:
        _add_warning("neither 'tokens' nor 'input_ids' present; cannot check total_lengths against token tensors")
    else:
        if not _is_seq(rollout_data[token_key]):
            _add_error(f"{token_key!r} must be list/tuple, got {type(rollout_data[token_key]).__name__}")
        elif len(rollout_data[token_key]) != n:
            _add_error(f"{token_key!r} length mismatch: got {len(rollout_data[token_key])}, expected {n}")

    if errors:
        _log("error", "summary_before_failure " + _basic_batch_summary())
        for w in warnings:
            _log("warning", w)
        for e in errors:
            _log("error", "failure " + e)
        raise ValueError(f"{where}: rollout validation failed with {len(errors)} error(s); see logs above")

    response_lengths = []
    total_lengths = []

    for i in range(n):
        # reward
        try:
            r = float(rollout_data["rewards"][i])
            if not math.isfinite(r):
                _add_error(f"rewards[{i}] is not finite: {r}")
        except Exception as e:
            _add_error(f"rewards[{i}] is not float-like: {type(e).__name__}: {e}")

        # lengths
        try:
            resp = int(rollout_data["response_lengths"][i])
            total = int(rollout_data["total_lengths"][i])
            response_lengths.append(resp)
            total_lengths.append(total)

            if resp <= 0:
                _add_error(f"response_lengths[{i}] must be > 0, got {resp}")
            if total <= 0:
                _add_error(f"total_lengths[{i}] must be > 0, got {total}")
            if resp > total:
                _add_error(f"response_lengths[{i}]={resp} > total_lengths[{i}]={total}")
        except Exception as e:
            _add_error(f"bad lengths at sample {i}: {type(e).__name__}: {e}")
            continue

        # tokens/input_ids
        if token_key is not None and _is_seq(rollout_data[token_key]) and i < len(rollout_data[token_key]):
            tok = rollout_data[token_key][i]
            tok_n = _numel(tok)
            if tok_n is None:
                _add_error(f"{token_key}[{i}] bad type: {type(tok).__name__}")
            elif tok_n != total:
                _add_error(f"{token_key}[{i}] length {tok_n} != total_lengths[{i}] {total}")
            if torch.is_tensor(tok) and tok.ndim != 1:
                _add_error(f"{token_key}[{i}] must be 1D, got shape={tuple(tok.shape)}")

        # masks
        mask = rollout_data["loss_masks"][i]
        mask_n = _numel(mask)
        mask_ndim = _ndim(mask)

        if mask_n is None:
            _add_error(f"loss_masks[{i}] bad type: {type(mask).__name__}")
            continue

        if mask_ndim != 1:
            _add_error(f"loss_masks[{i}] must be 1D, got ndim={mask_ndim}")

        if mask_n != resp:
            _add_error(f"loss_masks[{i}] length {mask_n} != response_lengths[{i}] {resp}")

        if not _finite_tensor_or_list(mask):
            _add_error(f"loss_masks[{i}] contains NaN/Inf or non-numeric values")
            continue

        mask_sum = _sum_float(mask)
        removed_by_filter = removed_by_filter_flags[i]
        mask_is_all_zero = _all_zero(mask)
        if removed_by_filter:
            if mask_is_all_zero:
                _add_warning(
                    f"loss_masks[{i}] has no active tokens because removed_by_filter=True, "
                    f"sum={mask_sum}, response_len={resp}"
                )
            else:
                _add_error(
                    f"loss_masks[{i}] is not all-zero despite removed_by_filter=True, "
                    f"sum={mask_sum}, response_len={resp}"
                )
        elif mask_sum <= 0:
            _add_error(
                f"loss_masks[{i}] has no active tokens without removed_by_filter=True, "
                f"sum={mask_sum}, response_len={resp}"
            )
        if mask_sum > resp:
            # Warning-only: float/weighted masks can legitimately have sum > resp.
            _add_warning(
                f"loss_masks[{i}] sum={mask_sum} exceeds response_len={resp} (expected for float/weighted masks)"
            )

        if torch.is_tensor(mask):
            # Binary check: warning, not fatal, because masks may be float.
            try:
                is_binary = bool(torch.all((mask == 0) | (mask == 1)).item())
                if not is_binary:
                    _add_warning(f"loss_masks[{i}] is not binary 0/1")
            except Exception as e:
                _add_warning(f"binary check failed for loss_masks[{i}]: {type(e).__name__}: {e}")

    # max_seq_lens if present.
    if _present("max_seq_lens"):
        xs = rollout_data["max_seq_lens"]
        if not _is_seq(xs):
            _add_error(f"max_seq_lens must be list/tuple, got {type(xs).__name__}")
        elif len(xs) != n:
            _add_error(f"max_seq_lens length {len(xs)} != expected {n}")
        else:
            for i, x in enumerate(xs):
                try:
                    msl = int(x)
                    if msl <= 0:
                        _add_error(f"max_seq_lens[{i}] must be > 0, got {msl}")
                    elif i < len(total_lengths) and msl < total_lengths[i]:
                        _add_error(f"max_seq_lens[{i}]={msl} < total_lengths[{i}]={total_lengths[i]}")
                except Exception as e:
                    _add_error(f"max_seq_lens[{i}] is not int-like: {type(e).__name__}: {e}")

    # Optional per-response vector fields.
    def check_vector_list(key, expected_lengths):
        if not _present(key):
            return
        xs = rollout_data[key]
        if not _is_seq(xs):
            _add_error(f"{key} must be list/tuple, got {type(xs).__name__}")
            return
        if len(xs) != n:
            _add_error(f"{key} length {len(xs)} != expected {n}")
            return

        for i, x in enumerate(xs):
            x_n = _numel(x)
            x_ndim = _ndim(x)
            if x_n is None:
                _add_error(f"{key}[{i}] bad type: {type(x).__name__}")
                continue
            if x_ndim != 1:
                _add_error(f"{key}[{i}] must be 1D, got ndim={x_ndim}")
            if i < len(expected_lengths) and x_n != expected_lengths[i]:
                _add_error(f"{key}[{i}] length {x_n} != response_lengths[{i}] {expected_lengths[i]}")
            if not _finite_tensor_or_list(x):
                _add_error(f"{key}[{i}] contains NaN/Inf or non-numeric values")

    logprob_key = "rollout_log_probs" if getattr(args, "use_rollout_logprobs", False) else "log_probs"

    if require_log_probs and not _present(logprob_key):
        _add_error(f"require_log_probs=True but {logprob_key!r} is missing/None")

    # rollout_log_probs are CP-sliced by get_rollout_data before this validator
    # runs, so their per-sample lengths differ from response_lengths[i] on any
    # rank when cp_size > 1. Skip the length check in that case.
    cp_size = int(_maybe_attr(ps, "cp", "size", default=1) or 1)
    for key in ("log_probs", "ref_log_probs", "values", "advantages", "returns"):
        check_vector_list(key, response_lengths)
    check_vector_list("rollout_log_probs", [] if cp_size > 1 else response_lengths)

    # This is important for your failure mode:
    # If compute_advantages_and_returns will normalize, every rank that reaches
    # it must have log_probs/values in the same structural state.
    if getattr(args, "compute_advantages_and_returns", False) and getattr(args, "normalize_advantages", False):
        if _present(logprob_key):
            _add_warning(
                f"normalization path will enter distributed whitening with {logprob_key}; "
                "if another rank is missing this key, it can skip or fail differently"
            )

    if warnings:
        for w in warnings[:50]:
            _log("warning", w)
        if len(warnings) > 50:
            _log("warning", f"... {len(warnings) - 50} additional warnings omitted")

    if errors:
        _log("error", "summary_before_failure " + _basic_batch_summary())
        for e in errors[:100]:
            _log("error", "failure " + e)
        if len(errors) > 100:
            _log("error", f"... {len(errors) - 100} additional errors omitted")
        _log("error", "trace_at_validation_failure\n" + "".join(traceback.format_stack(limit=12)))
        raise ValueError(f"{where}: rollout validation failed with {len(errors)} error(s); see logs above")

    _log("warning", "success " + _basic_batch_summary())


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref)

        init(args)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_main_rank = is_megatron_main_rank()

        if self._is_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay")
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "broadcast":
                update_weight_cls = UpdateWeightFromDistributed
            else:
                update_weight_cls = UpdateWeightP2P
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            is_lora=is_lora_enabled(args),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    def _fill_replay_data(
        self,
        data_iterator,
        num_microbatches,
        rollout_data,
        data_key: str,
        replay_list: list,
        register_replay_list_func,
        if_sp_region=True,
    ):
        if data_key not in rollout_data:
            raise ValueError(f"{data_key} is required in rollout_data for replay.")

        for iterator in data_iterator:
            iterator.reset()

        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size
        qkv_format = self.args.qkv_format

        def pad_func(data, pad):
            _, num_layers, topk = data.shape
            pad_tensor = torch.full(
                (pad, num_layers, topk),
                fill_value=-1,
                device=data.device,
                dtype=data.dtype,
            )
            return torch.cat([data, pad_tensor], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
            replay_data = batch[data_key]
            tokens = batch["tokens"]
            assert len(replay_data) == len(tokens)
            for a, b in zip(replay_data, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # TODO: maybe extract a common process function for here and get_batch?

            if qkv_format == "bshd":
                max_seqlen = batch["max_seq_lens"][0]
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
                replay_data = torch.stack(replay_data, dim=0)
                batch_size, seqlen, num_layers, topk = replay_data.shape
                replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                replay_data = torch.cat(replay_data, dim=0)
                pad_size = parallel_state.tp.size * self.args.data_pad_size_multiplier
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

            if self.args.sequence_parallel and if_sp_region:
                seqlen = replay_data.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                replay_data = replay_data[start:end]

            register_replay_list_func(replay_list, replay_data, self.model)

        del rollout_data[data_key]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        use_rollout_sampling_masks = "rollout_sampling_masks" in data_iterator[0].rollout_data
        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                use_rollout_sampling_masks=use_rollout_sampling_masks,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        validate_rollout_for_grpo_training_step(
            self.args,
            rollout_data,
            where=f"train_actor.rollout_id={rollout_id}.initial",
            logger=logger,
            require_log_probs=False,
        )

        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                self._fill_replay_data(
                    data_iterator,
                    num_microbatches,
                    rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=get_register_replay_list_func(m),
                    if_sp_region=m.if_sp_region,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        if self.args.offload_train and is_lora_enabled(self.args):
            # For LoRA, we must resume() to restore GPU memory backing for adapter
            # weights. Unlike base model weights (which are read from CPU backups),
            # LoRA adapter weights are accessed directly from GPU model parameters.
            # The disable() context alone only prevents new allocations from being
            # tracked -- it does NOT restore previously paused/offloaded tensors.
            torch_memory_saver.resume()
        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if not is_lora_enabled(self.args):
                if getattr(self.args, "check_all_engine_weight_versions", False):
                    assert_rollout_engine_weight_versions(
                        rollout_engines,
                        self.weight_updater.weight_version,
                    )
                elif self.args.ci_test and len(rollout_engines) > 0:
                    engine = random.choice(rollout_engines)
                    engine_version = ray.get(engine.get_weight_version.remote())
                    if str(engine_version) != str(self.weight_updater.weight_version):
                        raise RuntimeError(
                            f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                        )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            if is_lora_enabled(self.args):
                torch_memory_saver.pause()
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
