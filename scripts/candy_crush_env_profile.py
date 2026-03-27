from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pufferlib.ocean.candy_crush.candy_crush import CandyCrush


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Candy Crush env-side hot paths with random actions.")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=128)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    env = CandyCrush(
        num_envs=args.num_envs,
        render_mode="None",
        log_interval=args.log_interval,
        profile_perf=1,
    )
    try:
        env.reset(args.seed)
        rng = np.random.default_rng(args.action_seed)
        cache = 1024
        actions = rng.integers(
            0,
            env.single_action_space.n,
            size=(cache, env.num_agents),
            dtype=np.int32,
        )

        latest_info: dict[str, float] = {}
        steps = 0
        start = time.time()
        while time.time() - start < args.seconds:
            _, _, _, _, info = env.step(actions[steps % cache])
            steps += 1
            if info:
                latest_info = dict(info[-1])

        elapsed = time.time() - start
        result = {
            "num_envs": args.num_envs,
            "seconds": elapsed,
            "steps": steps,
            "sps": float(env.num_agents * steps / max(elapsed, 1e-9)),
            **latest_info,
        }
    finally:
        env.close()

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
