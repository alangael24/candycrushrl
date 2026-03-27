from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from pufferlib import pufferl


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_FILES = {
    "a0-taskdist": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_taskdist.ini",
    "a0-campaign": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_campaign.ini",
    "throughput": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_throughput.ini",
}
PRESET_ALIASES = {
    "b0-taskdist": "a0-taskdist",
    "b0-campaign": "a0-campaign",
}


def load_args_from_config(config_path: Path) -> dict[str, Any]:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config_file(str(config_path))
    finally:
        sys.argv = argv


def apply_overrides(args: dict[str, Any], cli: argparse.Namespace) -> dict[str, Any]:
    args = copy.deepcopy(args)

    if cli.device is not None:
        args["train"]["device"] = cli.device
    if cli.total_timesteps is not None:
        args["train"]["total_timesteps"] = cli.total_timesteps
    if cli.seed is not None:
        args["train"]["seed"] = cli.seed
    if cli.env_num_envs is not None:
        args["env"]["num_envs"] = cli.env_num_envs
    if cli.vec_num_envs is not None:
        args["vec"]["num_envs"] = cli.vec_num_envs
    if cli.tag is not None:
        args["tag"] = cli.tag
    if cli.load_model_path is not None:
        args["load_model_path"] = cli.load_model_path
    if cli.render_mode is not None:
        args["render_mode"] = cli.render_mode
    if cli.wandb:
        args["wandb"] = True
    if cli.neptune:
        args["neptune"] = True
    if cli.no_model_upload:
        args["no_model_upload"] = True

    return args


def resolve_preset(name: str) -> tuple[str, Path]:
    canonical = PRESET_ALIASES.get(name, name)
    if canonical not in PRESET_FILES:
        choices = ", ".join(sorted([*PRESET_FILES, *PRESET_ALIASES]))
        raise SystemExit(f"Unknown preset '{name}'. Valid presets: {choices}")

    return canonical, PRESET_FILES[canonical]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Candy Crush train/eval using explicit preset config files.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--preset",
            default="a0-taskdist",
            choices=sorted([*PRESET_FILES, *PRESET_ALIASES]),
            help="Named Candy Crush preset. 'b0-*' aliases point at the matching A0 config with different evaluation intent.",
        )
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--seed", type=int, default=None)
        subparser.add_argument("--total-timesteps", type=int, default=None)
        subparser.add_argument("--env-num-envs", type=int, default=None)
        subparser.add_argument("--vec-num-envs", type=int, default=None)
        subparser.add_argument("--tag", default=None)
        subparser.add_argument("--load-model-path", default=None)
        subparser.add_argument("--wandb", action="store_true")
        subparser.add_argument("--neptune", action="store_true")
        subparser.add_argument("--no-model-upload", action="store_true")

    train = subparsers.add_parser("train", help="Train Candy Crush with an explicit preset.")
    add_shared(train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate Candy Crush with an explicit preset.")
    add_shared(eval_parser)
    eval_parser.add_argument("--render-mode", default="human")

    dump = subparsers.add_parser("dump-config", help="Print the resolved nested config as JSON.")
    add_shared(dump)
    dump.add_argument("--render-mode", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()

    canonical, config_path = resolve_preset(cli.preset)
    args = load_args_from_config(config_path)
    args = apply_overrides(args, cli)

    if cli.mode == "dump-config":
        print(json.dumps({"preset": canonical, "config_path": str(config_path), "args": args}, indent=2, sort_keys=True))
        return

    env_name = args["env_name"]
    if cli.mode == "train":
        pufferl.train(env_name=env_name, args=args)
        return

    pufferl.eval(env_name=env_name, args=args)


if __name__ == "__main__":
    main()
