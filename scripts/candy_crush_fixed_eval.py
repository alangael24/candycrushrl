from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pufferlib.pytorch
from pufferlib import pufferl
from pufferlib.ocean.candy_crush.candy_crush import CandyCrush as CandyCrushEnv
from pufferlib.ocean.torch import CandyCrush as CandyCrushPolicy


PRESET_FILES = {
    "a0-taskdist": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_taskdist.ini",
    "a0-campaign": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_campaign.ini",
    "ingredient-pbrs-005": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_005.ini",
    "ingredient-pbrs-010": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_010.ini",
    "ingredient-pbrs-020": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_ingredient_pbrs_020.ini",
    "screen-200m": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_screen_200m.ini",
    "throughput": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_throughput.ini",
}
PRESET_ALIASES = {
    "b0-taskdist": "a0-taskdist",
    "b0-campaign": "a0-campaign",
    "ingredient-pbrs": "ingredient-pbrs-010",
    "screen-short": "screen-200m",
}

METRIC_KEYS = [
    "episode_return",
    "episode_length",
    "score",
    "goal_progress",
    "level_wins",
    "invalid_swaps",
    "successful_swaps",
    "frosting_cleared",
    "ingredient_dropped",
    "ingredient_progress_dense",
    "jelly_cleared",
    "color_collected",
]


def load_args_from_config(config_path: Path) -> dict[str, Any]:
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config_file(str(config_path))
    finally:
        sys.argv = argv


def resolve_preset(name: str) -> tuple[str, Path]:
    canonical = PRESET_ALIASES.get(name, name)
    if canonical not in PRESET_FILES:
        choices = ", ".join(sorted([*PRESET_FILES, *PRESET_ALIASES]))
        raise SystemExit(f"Unknown preset '{name}'. Valid presets: {choices}")
    return canonical, PRESET_FILES[canonical]


def summarize(episodes: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"episodes": len(episodes)}
    for key in METRIC_KEYS:
        values = [float(ep.get(key, 0.0)) for ep in episodes]
        summary[f"mean_{key}"] = float(sum(values) / len(values)) if values else 0.0
    return summary


def infer_reset_slices(env: CandyCrushEnv, obs: np.ndarray) -> dict[str, float]:
    flat = np.asarray(obs).reshape(-1).astype(np.uint8)
    feature_dim = flat.size - env.single_action_space.n
    board_dim = (env.num_candies * 5 + 4) * env.board_size * env.board_size
    goal_slots = env.num_candies + 4
    meta = flat[board_dim:feature_dim]
    targets = meta[:goal_slots]
    return {
        "task_active_goals": float(np.count_nonzero(targets)),
        "task_family_color": float(np.any(targets[:env.num_candies] > 0)),
        "task_family_jelly": float(targets[env.num_candies] > 0),
        "task_family_frosting": float(targets[env.num_candies + 1] > 0),
        "task_family_ingredient": float(targets[env.num_candies + 2] > 0),
        "task_family_score": float(targets[env.num_candies + 3] > 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed Candy Crush evaluation split.")
    parser.add_argument("--preset", default="screen-200m", choices=sorted([*PRESET_FILES, *PRESET_ALIASES]))
    parser.add_argument("--load-model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--action-seed", type=int, default=20260327)
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    _, config_path = resolve_preset(cli.preset)
    args = copy.deepcopy(load_args_from_config(config_path))

    torch.manual_seed(cli.action_seed)
    np.random.seed(cli.action_seed)

    env_kwargs = dict(args["env"])
    env_kwargs["num_envs"] = 1
    env_kwargs["render_mode"] = "None"
    env_kwargs["log_interval"] = 1

    env = CandyCrushEnv(**env_kwargs)
    policy = CandyCrushPolicy(env).to(cli.device)
    state = torch.load(cli.load_model_path, map_location=cli.device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    policy.load_state_dict(state)
    policy.eval()

    completed: list[dict[str, float]] = []
    total_steps = 0

    for episode_index in range(cli.episodes):
        obs, _ = env.reset(seed=cli.seed_start + episode_index)
        current = defaultdict(float)
        current.update(infer_reset_slices(env, obs))
        current_len = 0

        while True:
            obs_t = torch.as_tensor(obs).to(cli.device)
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_t)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)

            obs, reward, terminal, truncated, info = env.step(int(np.asarray(action).reshape(-1)[0]))
            current["episode_return"] += float(np.asarray(reward).reshape(-1)[0])
            current_len += 1
            total_steps += 1

            if info:
                log = info[-1]
                if isinstance(log, dict):
                    for key, value in log.items():
                        if key != "n":
                            current[key] = float(value)

            done = bool(np.asarray(terminal).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
            if done:
                current["episode_length"] = current_len
                completed.append(dict(current))
                break

    family_preds = {
        "color": lambda ep: ep.get("task_family_color", 0.0) > 0.5,
        "jelly": lambda ep: ep.get("task_family_jelly", 0.0) > 0.5,
        "frosting": lambda ep: ep.get("task_family_frosting", 0.0) > 0.5,
        "ingredient": lambda ep: ep.get("task_family_ingredient", 0.0) > 0.5,
        "score": lambda ep: ep.get("task_family_score", 0.0) > 0.5,
    }

    active_goal_rows = {}
    for n in (1, 2, 3):
        bucket = [ep for ep in completed if int(ep.get("task_active_goals", 0.0)) == n]
        active_goal_rows[str(n)] = summarize(bucket)

    family_rows = {}
    for name, pred in family_preds.items():
        bucket = [ep for ep in completed if pred(ep)]
        family_rows[name] = summarize(bucket)

    report = {
        "checkpoint": cli.load_model_path,
        "device": cli.device,
        "episodes": len(completed),
        "seed_start": cli.seed_start,
        "action_seed": cli.action_seed,
        "steps": total_steps,
        "overall": summarize(completed),
        "task_active_goals": active_goal_rows,
        "families": family_rows,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if cli.json_out is not None:
        Path(cli.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
