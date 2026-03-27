from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pufferlib import pufferl


DEFAULT_LAYOUTS = ("64x32", "128x16", "256x8", "512x4")
SUMMARY_KEYS = (
    "SPS",
    "environment/episode_return",
    "environment/level_wins",
    "environment/goal_progress",
    "losses/explained_variance",
    "performance/eval",
    "performance/env",
    "performance/eval_copy",
    "performance/eval_forward",
    "performance/train",
    "performance/train_copy",
    "performance/train_forward",
    "environment/perf_resolve_ms",
    "environment/perf_obs_ms",
    "environment/perf_board_obs_ms",
    "environment/perf_meta_obs_ms",
    "environment/perf_action_mask_ms",
)


def load_base_args(env_name: str) -> dict[str, Any]:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(env_name)
    finally:
        sys.argv = argv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layout(layout: str) -> tuple[int, int]:
    left, right = layout.lower().split("x", 1)
    return int(left), int(right)


def apply_a0_layout(
    args: dict[str, Any],
    *,
    device: str,
    total_timesteps: int,
    data_dir: str,
    env_num: int,
    vec_num: int,
    seed: int,
    profile_perf: bool,
) -> dict[str, Any]:
    args = copy.deepcopy(args)
    args["wandb"] = False
    args["neptune"] = False
    args["no_model_upload"] = True

    args["env"]["num_envs"] = env_num
    args["env"]["profile_perf"] = int(profile_perf)

    args["vec"]["num_envs"] = vec_num
    args["vec"]["num_workers"] = vec_num
    args["vec"]["batch_size"] = vec_num

    args["train"]["device"] = device
    args["train"]["seed"] = seed
    args["train"]["total_timesteps"] = total_timesteps
    args["train"]["minibatch_size"] = 65536
    args["train"]["max_minibatch_size"] = 65536
    args["train"]["update_epochs"] = 4
    args["train"]["learning_rate"] = 0.01
    args["train"]["optimizer"] = "adam"
    args["train"]["data_dir"] = data_dir
    return args


def summarize_logs(logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        return {}

    final = logs[-1]
    for item in reversed(logs):
        if "SPS" in item:
            final = item
            break

    summary: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        if key in final:
            summary[key] = float(final[key])
    if "agent_steps" in final:
        summary["agent_steps"] = int(final["agent_steps"])
    if "epoch" in final:
        summary["epoch"] = int(final["epoch"])
    return summary


def run_layout(
    env_name: str,
    base_args: dict[str, Any],
    *,
    layout: str,
    device: str,
    total_timesteps: int,
    seed: int,
    profile_perf: bool,
    scratch_root: Path | None,
) -> dict[str, Any]:
    env_num, vec_num = parse_layout(layout)
    seed_everything(seed)
    tmpdir_obj = tempfile.TemporaryDirectory(
        prefix=f"cc_geom_{env_num}x{vec_num}_",
        dir=None if scratch_root is None else str(scratch_root),
    )
    try:
        args = apply_a0_layout(
            base_args,
            device=device,
            total_timesteps=total_timesteps,
            data_dir=tmpdir_obj.name,
            env_num=env_num,
            vec_num=vec_num,
            seed=seed,
            profile_perf=profile_perf,
        )
        logger = pufferl.NoLogger(args)
        start = time.time()
        logs = pufferl.train(env_name=env_name, args=args, logger=logger)
        elapsed = time.time() - start
    finally:
        tmpdir_obj.cleanup()

    summary = summarize_logs(logs)
    summary.update(
        {
            "layout": layout,
            "env_num_envs": env_num,
            "vec_num_envs": vec_num,
            "wall_time_s": elapsed,
            "profile_perf": bool(profile_perf),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Candy Crush A0 geometry on a single host.")
    parser.add_argument("--env-name", default="puffer_candy_crush")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--layouts", nargs="+", default=list(DEFAULT_LAYOUTS))
    parser.add_argument("--profile-perf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    base_args = load_base_args(args.env_name)
    results = []
    for layout in args.layouts:
        result = run_layout(
            args.env_name,
            base_args,
            layout=layout,
            device=args.device,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            profile_perf=args.profile_perf,
            scratch_root=args.scratch_root,
        )
        results.append(result)
        print(
            f"{layout}: SPS={result.get('SPS', 0.0):.1f} "
            f"env={result.get('performance/env', 0.0):.3f} "
            f"copy={result.get('performance/eval_copy', 0.0):.3f} "
            f"forward={result.get('performance/eval_forward', 0.0):.3f}"
        )

    payload = {
        "env_name": args.env_name,
        "device": args.device,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "profile_perf": bool(args.profile_perf),
        "layouts": list(args.layouts),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
