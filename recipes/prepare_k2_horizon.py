"""Prepare a public K2 Horizon 7B download for the bundled RL360 runtime."""

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile


def prepare_checkpoint(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output already exists; choose a new directory: {destination}")

    config = json.loads((source / "config.json").read_text())
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    expected = {
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layernorm_num_groups": 4,
        "vocab_size": 250624,
        "query_key_norm": False,
        "num_experts": 0,
        "mova_num_experts": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Expected K2 Horizon 7B {key}={value!r}, got {config.get(key)!r}")
    rope = config.get("rope_parameters") or {}
    if rope.get("rope_type") != "default" or rope.get("rope_theta") != 10000000:
        raise ValueError("This recipe expects K2 Horizon 7B with default RoPE and base 10000000")
    if tokenizer_config.get("tokenizer_class") not in ("TokenizersBackend", "PreTrainedTokenizerFast"):
        raise ValueError("Expected the public K2 Horizon fast tokenizer")

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = set(index["weight_map"].values())
    if not weights:
        raise ValueError("The checkpoint index contains no weight shards")
    for name in ["tokenizer.json", "chat_template.jinja", *weights]:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Missing checkpoint file or unsupported nested filename: {name}")

    config["auto_map"] = {
        "AutoConfig": "configuration_rl360_k2_horizon.K2HorizonConfig",
    }
    config["torch_dtype"] = config.get("torch_dtype", config.get("dtype", "bfloat16"))
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    replacements = {"config.json": config, "tokenizer_config.json": tokenizer_config}
    shim = Path(__file__).with_name("configuration_rl360_k2_horizon.py")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Keep the original download intact. Hard links avoid another copy of the
    # weights on shared storage; separate files hold the adapted metadata.
    with tempfile.TemporaryDirectory(prefix=".rl360-k2-", dir=destination.parent) as temporary:
        output = Path(temporary)
        for path in source.iterdir():
            if not path.is_file() or path.name in replacements or path.name == shim.name:
                continue
            try:
                os.link(path, output / path.name)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(path, output / path.name)
        for name, data in replacements.items():
            (output / name).write_text(json.dumps(data, indent=2) + "\n")
        shutil.copy2(shim, output / shim.name)
        output.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory downloaded from IFM/K2-Horizon-7B")
    parser.add_argument("destination", type=Path, help="New directory for the RL360 checkpoint")
    args = parser.parse_args()
    try:
        prepare_checkpoint(args.source, args.destination)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Checkpoint preparation failed: {error}\n")
    print(f"Prepared K2 Horizon 7B checkpoint: {args.destination}")


if __name__ == "__main__":
    main()
