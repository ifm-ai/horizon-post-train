"""Shared helpers for the agent360.harbor.miles package."""


def coerce_reward_to_float(value) -> float:
    """Coerce a possibly-None reward to a finite float.

    Harbor agent server JSON returns ``{"reward": null}`` on some failure
    paths (radixark/miles#989). Python's ``dict.get(key, default)`` does
    not coerce a present-but-None value, so callers must guard explicitly
    before passing to ``torch.tensor(..., dtype=torch.float)``.
    """
    return 0.0 if value is None else float(value)
