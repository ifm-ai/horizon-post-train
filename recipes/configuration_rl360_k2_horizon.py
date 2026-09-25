"""Load the public K2 Horizon config with the bundled Transformers version."""

from transformers import PretrainedConfig


class K2HorizonConfig(PretrainedConfig):
    model_type = "k2_horizon"

    def __init__(self, **kwargs):
        # Architecture fields come from the downloaded config.json. The public
        # config class uses a newer Transformers dataclass API; the training
        # stack only needs the same fields exposed through PretrainedConfig.
        super().__init__(**kwargs)
        rope = getattr(self, "rope_parameters", None) or {}
        self.rope_theta = rope.get("rope_theta", getattr(self, "rope_theta", 10000))
        self.rope_scaling = None if rope.get("rope_type", "default") == "default" else rope
