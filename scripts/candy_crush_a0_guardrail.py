from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pufferlib import pufferl


DEFAULT_SEEDS = (101, 202, 303)
METRICS = (
    "environment/episode_return",
    "environment/level_wins",
    "environment/goal_progress",
    "losses/explained_variance",
)


def load_base_args(env_name: str) -> dict[str, Any]:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(env_name)
    finally:
        sys.argv = argv


def apply_a0(args: dict[str, Any], device: str, total_timesteps: int, data_dir: str) -> dict[str, Any]:
    args = copy.deepcopy(args)
    args["wandb"] = False
    args["neptune"] = False
    args["no_model_upload"] = True

    args["env"]["num_envs"] = 256
    args["vec"]["num_envs"] = 8

    args["train"]["device"] = device
    args["train"]["total_timesteps"] = total_timesteps
    args["train"]["minibatch_size"] = 65536
    args["train"]["max_minibatch_size"] = 65536
    args["train"]["update_epochs"] = 4
    args["train"]["learning_rate"] = 0.01
    args["train"]["optimizer"] = "adam"
    args["train"]["data_dir"] = data_dir
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_series(logs: list[dict[str, Any]]) -> list[dict[str, float]]:
    series: list[dict[str, float]] = []
    for log in logs:
        if not log or "agent_steps" not in log:
            continue

        point: dict[str, float] = {"agent_steps": float(log["agent_steps"])}
        for metric in METRICS:
            if metric in log:
                point[metric] = float(log[metric])

        if len(point) > 1:
            series.append(point)

    series.sort(key=lambda point: point["agent_steps"])
    return series


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not runs:
        return []

    step_sets = []
    for run in runs:
        step_sets.append({int(point["agent_steps"]) for point in run["series"]})

    common_steps = sorted(set.intersection(*step_sets)) if step_sets else []
    aggregate: list[dict[str, float]] = []
    for step in common_steps:
        point: dict[str, float] = {"agent_steps": float(step)}
        for metric in METRICS:
            values = [
                float(next(item[metric] for item in run["series"] if int(item["agent_steps"]) == step))
                for run in runs
            ]
            point[f"{metric}_mean"] = float(np.mean(values))
            point[f"{metric}_std"] = float(np.std(values))
        aggregate.append(point)
    return aggregate


def run_guardrail(env_name: str, seeds: list[int], device: str, total_timesteps: int, scratch_root: Path | None) -> dict[str, Any]:
    base_args = load_base_args(env_name)
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        seed_everything(seed)
        tmpdir_obj = tempfile.TemporaryDirectory(
            prefix=f"cc_a0_guardrail_{seed}_",
            dir=None if scratch_root is None else str(scratch_root),
        )
        try:
            args = apply_a0(
                base_args,
                device=device,
                total_timesteps=total_timesteps,
                data_dir=tmpdir_obj.name,
            )
            args["train"]["seed"] = seed
            logger = pufferl.NoLogger(args)
            logs = pufferl.train(env_name=env_name, args=args, logger=logger)
            series = extract_series(logs)
            runs.append(
                {
                    "seed": seed,
                    "series": series,
                    "final": series[-1] if series else {},
                }
            )
        finally:
            tmpdir_obj.cleanup()

    aggregate = aggregate_runs(runs)
    return {
        "env_name": env_name,
        "device": device,
        "total_timesteps": total_timesteps,
        "seeds": seeds,
        "metrics": list(METRICS),
        "runs": runs,
        "aggregate": {
            "series": aggregate,
            "final": aggregate[-1] if aggregate else {},
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def compare_reports(expected: dict[str, Any], actual: dict[str, Any], atol: float, rtol: float) -> None:
    expected_series = {int(point["agent_steps"]): point for point in expected["aggregate"]["series"]}
    actual_series = {int(point["agent_steps"]): point for point in actual["aggregate"]["series"]}
    common_steps = sorted(set(expected_series) & set(actual_series))
    if not common_steps:
        raise AssertionError("No overlapping agent_steps between baseline and current run")

    failures = []
    for step in common_steps:
        for metric in METRICS:
            key = f"{metric}_mean"
            expected_value = float(expected_series[step][key])
            actual_value = float(actual_series[step][key])
            if not np.isclose(actual_value, expected_value, atol=atol, rtol=rtol):
                failures.append((step, metric, expected_value, actual_value))

    if failures:
        lines = ["A0 learning guardrail regression detected:"]
        for step, metric, expected_value, actual_value in failures[:16]:
            lines.append(
                f"- step {step} {metric}: expected {expected_value:.6f}, got {actual_value:.6f}"
            )
        if len(failures) > 16:
            lines.append(f"- ... and {len(failures) - 16} more mismatches")
        raise AssertionError("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Candy Crush A0 learning guardrail.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record = subparsers.add_parser("record", help="Run A0 on multiple seeds and save the baseline report.")
    record.add_argument("report_path", type=Path)
    record.add_argument("--env-name", default="puffer_candy_crush")
    record.add_argument("--device", default="cuda")
    record.add_argument("--total-timesteps", type=int, default=10_000_000)
    record.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    record.add_argument("--scratch-root", type=Path, default=None)

    check = subparsers.add_parser("check", help="Run A0 and compare to a saved baseline report.")
    check.add_argument("report_path", type=Path)
    check.add_argument("--env-name", default="puffer_candy_crush")
    check.add_argument("--device", default="cuda")
    check.add_argument("--total-timesteps", type=int, default=10_000_000)
    check.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    check.add_argument("--scratch-root", type=Path, default=None)
    check.add_argument("--atol", type=float, default=0.02)
    check.add_argument("--rtol", type=float, default=0.05)
    check.add_argument("--write-current", type=Path, default=None)

    args = parser.parse_args()
    report = run_guardrail(
        env_name=args.env_name,
        seeds=list(args.seeds),
        device=args.device,
        total_timesteps=args.total_timesteps,
        scratch_root=args.scratch_root,
    )

    if args.mode == "record":
        write_json(args.report_path, report)
        print(f"Recorded A0 learning baseline to {args.report_path}")
        return

    with args.report_path.open("r", encoding="utf-8") as f:
        expected = json.load(f)

    if args.write_current is not None:
        write_json(args.write_current, report)

    compare_reports(expected, report, atol=args.atol, rtol=args.rtol)
    print(f"A0 learning guardrail passed: {args.report_path}")


if __name__ == "__main__":
    main()
