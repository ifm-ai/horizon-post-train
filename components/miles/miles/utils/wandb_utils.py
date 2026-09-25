import logging
import os
import wandb

logger = logging.getLogger(__name__)

# Log experiment settings deliberately. The full argument namespace can contain
# API credentials, checkpoint paths, service URLs and encoded environment data.
_PUBLIC_CONFIG_FIELDS = frozenset(
    "actor_num_nodes actor_num_gpus_per_node rollout_num_gpus rollout_num_gpus_per_engine "
    "num_rollout rollout_batch_size n_samples_per_prompt global_batch_size micro_batch_size "
    "use_dynamic_global_batch_size use_dynamic_batch_size rollout_max_response_len "
    "rollout_temperature rollout_top_p rollout_top_k sglang_context_length "
    "tensor_model_parallel_size context_parallel_size pipeline_model_parallel_size "
    "expert_model_parallel_size expert_tensor_parallel_size sequence_parallel "
    "num_layers hidden_size ffn_hidden_size num_attention_heads num_query_groups "
    "kv_channels vocab_size max_position_embeddings rotary_base norm_epsilon "
    "group_query_attention swiglu qk_layernorm untie_embeddings_and_output_weights "
    "lr min_lr lr_warmup_iters lr_warmup_fraction adam_beta1 adam_beta2 adam_eps "
    "weight_decay clip_grad eps_clip eps_clip_high entropy_coef kl_coef kl_loss_coef "
    "grpo_std_normalization use_tis tis_clip tis_clip_low use_kl_loss "
    "attention_dropout hidden_dropout seed bf16 fp16 max_tokens_per_gpu "
    "save_interval update_weights_interval optimizer_cpu_offload keep_old_actor"
    .split()
)
_PUBLIC_CONFIG_CHOICES = {
    "optimizer": {"adam", "sgd", "muon"},
    "advantage_estimator": {"grpo", "ppo", "reinforce_plus_plus", "rloo", "reinforce_plus_plus_baseline"},
    "lr_decay_style": {"constant", "linear", "cosine", "inverse-square-root", "WSD"},
    "kl_loss_type": {"kl", "abs", "mse", "low_var_kl", "full"},
}


def _wandb_settings(**kwargs):
    # Keep source, machine metadata and local console output out of run uploads.
    return wandb.Settings(disable_git=True, save_code=False, x_disable_meta=True, console="off", **kwargs)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = _wandb_settings(mode="offline")
    else:
        init_kwargs["settings"] = _wandb_settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    values = vars(args)
    output = {
        key: value for key, value in values.items()
        if key in _PUBLIC_CONFIG_FIELDS and (value is None or type(value) in (bool, int, float))
    }
    output.update({key: values[key] for key, choices in _PUBLIC_CONFIG_CHOICES.items()
                   if isinstance(values.get(key), str) and values[key] in choices})
    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_config_for_logging(args),
        "resume": "allow",
        "reinit": True,
        "settings": _wandb_settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    # `train/rollout_id` and `train/step_in_rollout` are alternative x-axes
    # to `train/step`. They let users view train/* metrics against a
    # stable rollout-cycle counter or per-rollout-step index rather than
    # the cumulative optimizer-step count, whose inter-rollout gaps are
    # not constant under dynamic batching.
    wandb.define_metric("train/rollout_id", step_metric="train/step")
    wandb.define_metric("train/step_in_rollout", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    # train/rollout_id is co-logged on both rollout-side and train-side log() calls
    # so that any train/* or rollout/* metric can be cross-plotted against the
    # rollout counter. Declared after the "train/*" wildcard so the specific name
    # isn't inadvertently treated as step-metric'd against train/step.
    wandb.define_metric("train/rollout_id")
    # Bare step counters — co-logged so one panel can plot all three as time series
    # (useful for spotting non-monotone train/step jumps under dynamic batching).
    wandb.define_metric("samples_seen", step_metric="rollout/step")
    wandb.define_metric("train_step", step_metric="train/step")
    wandb.define_metric("rollout_step", step_metric="rollout/step")
    # Bare reward/response_stats mirrors — stripped of the rollout/ prefix by
    # _log_rollout_data so the panels appear at the top level in W&B.
    wandb.define_metric("reward/*", step_metric="rollout/step")
    wandb.define_metric("response_stats/*", step_metric="rollout/step")
