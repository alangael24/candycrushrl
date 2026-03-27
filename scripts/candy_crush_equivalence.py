from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pufferlib import pufferl
from pufferlib.ocean.candy_crush.candy_crush import CandyCrush


def load_env_config(env_name: str) -> dict[str, Any]:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = argv
    return dict(args["env"])


def normalize_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_info(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_info(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_info(info: Any) -> str:
    return json.dumps(normalize_info(info), sort_keys=True, separators=(",", ":"))


def make_env(env_name: str, seed: int, num_envs: int, log_interval: int) -> tuple[CandyCrush, dict[str, Any]]:
    env_kwargs = load_env_config(env_name)
    env_kwargs.update(
        num_envs=num_envs,
        log_interval=log_interval,
        render_mode="None",
    )
    env = CandyCrush(seed=seed, **env_kwargs)
    return env, env_kwargs


def record_trace(trace_path: Path, env_name: str, seed: int, action_seed: int, steps: int, num_envs: int, log_interval: int) -> None:
    env, env_kwargs = make_env(env_name, seed, num_envs, log_interval)
    try:
        initial_obs, _ = env.reset(seed=seed)
        initial_obs = np.array(initial_obs, copy=True)
        actions = np.random.default_rng(action_seed).integers(
            0,
            env.single_action_space.n,
            size=(steps, num_envs),
            dtype=np.int32,
        )

        observations = np.empty((steps, *initial_obs.shape), dtype=initial_obs.dtype)
        rewards = np.empty((steps, num_envs), dtype=np.float32)
        terminals = np.empty((steps, num_envs), dtype=bool)
        truncations = np.empty((steps, num_envs), dtype=bool)
        infos: list[str] = []

        for step in range(steps):
            obs, rew, done, trunc, info = env.step(actions[step])
            observations[step] = np.array(obs, copy=True)
            rewards[step] = np.array(rew, copy=True)
            terminals[step] = np.array(done, copy=True)
            truncations[step] = np.array(trunc, copy=True)
            infos.append(serialize_info(info))

        metadata = {
            "env_name": env_name,
            "seed": seed,
            "action_seed": action_seed,
            "steps": steps,
            "num_envs": num_envs,
            "log_interval": log_interval,
            "env_kwargs": env_kwargs,
        }

        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            initial_observations=initial_obs,
            actions=actions,
            observations=observations,
            rewards=rewards,
            terminals=terminals,
            truncations=truncations,
            infos=np.array(infos),
        )
        print(f"Recorded Candy Crush trace to {trace_path}")
    finally:
        env.close()


def compare_array(name: str, expected: np.ndarray, actual: np.ndarray, step: int | None = None, atol: float = 1e-7) -> None:
    if expected.dtype.kind in "fc":
        equal = np.allclose(expected, actual, atol=atol, rtol=0.0)
    else:
        equal = np.array_equal(expected, actual)

    if equal:
        return

    where = f" at step {step}" if step is not None else ""
    raise AssertionError(f"{name} mismatch{where}")


def check_trace(trace_path: Path) -> None:
    with np.load(trace_path) as trace:
        metadata = json.loads(str(trace["metadata"]))
        expected_initial_obs = trace["initial_observations"]
        actions = trace["actions"]
        expected_observations = trace["observations"]
        expected_rewards = trace["rewards"]
        expected_terminals = trace["terminals"]
        expected_truncations = trace["truncations"]
        expected_infos = trace["infos"]

    env = CandyCrush(seed=metadata["seed"], **metadata["env_kwargs"])
    try:
        actual_initial_obs, _ = env.reset(seed=metadata["seed"])
        compare_array("initial_observations", expected_initial_obs, np.array(actual_initial_obs, copy=True))

        for step in range(int(metadata["steps"])):
            obs, rew, done, trunc, info = env.step(actions[step])
            compare_array("observations", expected_observations[step], np.array(obs, copy=True), step=step)
            compare_array("rewards", expected_rewards[step], np.array(rew, copy=True), step=step)
            compare_array("terminals", expected_terminals[step], np.array(done, copy=True), step=step)
            compare_array("truncations", expected_truncations[step], np.array(trunc, copy=True), step=step)

            actual_info = serialize_info(info)
            if str(expected_infos[step]) != actual_info:
                raise AssertionError(f"info mismatch at step {step}")

        print(f"Trace matches: {trace_path}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Candy Crush golden-trace guardrail.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record = subparsers.add_parser("record", help="Record a new golden trace.")
    record.add_argument("trace_path", type=Path)
    record.add_argument("--env-name", default="puffer_candy_crush")
    record.add_argument("--seed", type=int, default=1234)
    record.add_argument("--action-seed", type=int, default=2026)
    record.add_argument("--steps", type=int, default=4096)
    record.add_argument("--num-envs", type=int, default=1)
    record.add_argument("--log-interval", type=int, default=1)

    check = subparsers.add_parser("check", help="Replay and compare against a golden trace.")
    check.add_argument("trace_path", type=Path)

    args = parser.parse_args()
    if args.mode == "record":
        record_trace(
            trace_path=args.trace_path,
            env_name=args.env_name,
            seed=args.seed,
            action_seed=args.action_seed,
            steps=args.steps,
            num_envs=args.num_envs,
            log_interval=args.log_interval,
        )
    else:
        check_trace(args.trace_path)


if __name__ == "__main__":
    main()
