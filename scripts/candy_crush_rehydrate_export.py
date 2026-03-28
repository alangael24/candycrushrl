from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from pufferlib import pufferl


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_FILES = {
    "a0-taskdist": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_taskdist.ini",
    "a0-campaign": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_campaign.ini",
    "ingredient-pbrs-005": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_005.ini",
    "ingredient-pbrs-010": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_010.ini",
    "ingredient-pbrs-020": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_020.ini",
    "throughput": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_throughput.ini",
}
PRESET_ALIASES = {
    "b0-taskdist": "a0-taskdist",
    "b0-campaign": "a0-campaign",
    "ingredient-pbrs": "ingredient-pbrs-010",
}


def resolve_preset(name: str) -> tuple[str, Path]:
    canonical = PRESET_ALIASES.get(name, name)
    if canonical not in PRESET_FILES:
        choices = ", ".join(sorted([*PRESET_FILES, *PRESET_ALIASES]))
        raise SystemExit(f"Unknown preset '{name}'. Valid presets: {choices}")

    return canonical, PRESET_FILES[canonical]


def load_args_from_config(config_path: Path) -> dict:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config_file(str(config_path))
    finally:
        sys.argv = argv


def instantiate_policy(args: dict):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(args["env_name"])

    env_kwargs = copy.deepcopy(args["env"])
    env_kwargs["num_envs"] = 1
    env_kwargs["render_mode"] = "None"

    env = make_env(buf=None, seed=args["train"]["seed"], **env_kwargs)

    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(env, **args["policy"])

    rnn_name = args["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, rnn_name)
        policy = rnn_cls(env, policy, **args["rnn"])

    return env, policy


def rehydrate(weights_bin: Path, policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    flat = np.fromfile(weights_bin, dtype=np.float32)
    expected = sum(param.numel() for _, param in policy.named_parameters())
    if flat.size != expected:
        raise ValueError(
            f"Weight count mismatch for {weights_bin}: file has {flat.size} float32 values, "
            f"but policy expects {expected}."
        )

    offset = 0
    state_dict: dict[str, torch.Tensor] = {}
    for name, param in policy.named_parameters():
        count = param.numel()
        block = flat[offset : offset + count].reshape(tuple(param.shape))
        state_dict[name] = torch.from_numpy(block).clone().to(dtype=param.dtype)
        offset += count

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehydrate a Candy Crush exported weights .bin into a PyTorch .pt state_dict.")
    parser.add_argument("weights_bin", type=Path)
    parser.add_argument("--preset", default="b0-taskdist", choices=sorted([*PRESET_FILES, *PRESET_ALIASES]))
    parser.add_argument("--pt-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    canonical, config_path = resolve_preset(args.preset)
    config = load_args_from_config(config_path)
    config["train"]["seed"] = args.seed

    env = None
    try:
        env, policy = instantiate_policy(config)
        state_dict = rehydrate(args.weights_bin, policy)
        policy.load_state_dict(state_dict, strict=True)

        pt_out = args.pt_out
        if pt_out is None:
            pt_out = args.weights_bin.with_suffix(".pt")

        torch.save(policy.state_dict(), pt_out)
        print(f"Preset: {canonical}")
        print(f"Saved reconstructed checkpoint to {pt_out}")
        print(f"Parameters: {sum(param.numel() for _, param in policy.named_parameters())}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
