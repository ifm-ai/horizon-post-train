from argparse import Namespace


def validate_rollout_temperature(args: Namespace) -> None:
    """Reject temperatures that make training log-prob recomputation undefined."""
    if args.rollout_temperature < 0:
        raise ValueError("--rollout-temperature must be non-negative for generation.")
    if args.rollout_temperature == 0 and not args.debug_rollout_only:
        raise ValueError(
            "--rollout-temperature must be greater than 0 for training because "
            "Miles divides policy logits by it during log-probability recomputation. "
            "Temperature 0 is only supported with --debug-rollout-only for greedy generation."
        )
