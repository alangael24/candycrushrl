from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import pufferlib.pytorch
from pufferlib import pufferl


PRESET_FILES = {
    "a0-taskdist": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_taskdist.ini",
    "a0-campaign": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_a0_campaign.ini",
    "hardmix-ft": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_hardmix_ft.ini",
    "throughput": REPO_ROOT / "pufferlib" / "config" / "ocean" / "candy_crush_throughput.ini",
}
PRESET_ALIASES = {
    "b0-taskdist": "a0-taskdist",
    "b0-campaign": "a0-campaign",
    "b1-hardmix-ft": "hardmix-ft",
}
SUMMARY_KEYS = (
    "episode_return",
    "score",
    "goal_progress",
    "level_wins",
    "episode_length",
    "invalid_swaps",
)
TASK_FAMILY_KEYS = (
    ("task_family_color", "color"),
    ("task_family_jelly", "jelly"),
    ("task_family_frosting", "frosting"),
    ("task_family_ingredient", "ingredient"),
    ("task_family_score", "score"),
)


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
    if getattr(cli, "render_mode", None) is not None:
        args["render_mode"] = cli.render_mode
    if getattr(cli, "save_frames", None) is not None:
        args["save_frames"] = cli.save_frames
    if getattr(cli, "gif_path", None) is not None:
        args["gif_path"] = cli.gif_path
    if getattr(cli, "fps", None) is not None:
        args["fps"] = cli.fps
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


def build_env_module(args: dict[str, Any]):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    return importlib.import_module(module_name)


def make_eval_env(args: dict[str, Any]):
    env_module = build_env_module(args)
    make_env = env_module.env_creator(args["env_name"])
    env_kwargs = copy.deepcopy(args["env"])
    env_kwargs["num_envs"] = 1
    env_kwargs["log_interval"] = 1
    env_kwargs["render_mode"] = args.get("render_mode", "human")
    return make_env(**env_kwargs)


def family_label(episode: dict[str, float]) -> str:
    parts = [label for key, label in TASK_FAMILY_KEYS if float(episode.get(key, 0.0)) > 0.5]
    return "+".join(parts) if parts else "none"


def step_budget_bucket(step_budget: int, width: int) -> str:
    if step_budget <= 0:
        return "unknown"
    width = max(1, width)
    start = (step_budget // width) * width
    end = start + width - 1
    return f"{start:02d}-{end:02d}"


def summarize_episodes(episodes: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"episodes": len(episodes)}
    for key in SUMMARY_KEYS:
        values = [float(episode.get(key, 0.0)) for episode in episodes]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
    return summary


def build_slice_rows(
    episodes: list[dict[str, float]],
    slice_name: str,
    key_fn,
) -> list[dict[str, float | str]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for episode in episodes:
        buckets[str(key_fn(episode))].append(episode)

    rows: list[dict[str, float | str]] = []
    total = max(1, len(episodes))
    for slice_value in sorted(buckets, key=lambda value: (value.isdigit(), value)):
        bucket = buckets[slice_value]
        row: dict[str, float | str] = {
            "slice_name": slice_name,
            "slice_value": slice_value,
            "episodes": len(bucket),
            "share": len(bucket) / total,
        }
        row.update(summarize_episodes(bucket))
        rows.append(row)

    rows.sort(key=lambda row: (-int(row["episodes"]), str(row["slice_value"])))
    return rows


def print_slice_rows(title: str, rows: list[dict[str, float | str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  no episodes")
        return

    for row in rows:
        print(
            "  "
            f"{row['slice_value']}: "
            f"n={int(row['episodes'])} "
            f"share={float(row['share']):.2%} "
            f"return={float(row['mean_episode_return']):.3f} "
            f"goal={float(row['mean_goal_progress']):.3f} "
            f"wins={float(row['mean_level_wins']):.3f} "
            f"len={float(row['mean_episode_length']):.2f} "
            f"score={float(row['mean_score']):.2f}"
        )


def run_candy_crush_eval(args: dict[str, Any], episodes: int, step_bin_width: int) -> dict[str, Any]:
    env = make_eval_env(args)
    policy = pufferl.load_policy(args, SimpleNamespace(driver_env=env), args["env_name"])
    device = args["train"]["device"]
    torch.manual_seed(int(args["train"].get("seed", 0)))
    render_enabled = str(args.get("render_mode", "human")).lower() != "none"

    state = {}
    if args["train"].get("use_rnn"):
        state = dict(
            lstm_h=torch.zeros(1, policy.hidden_size, device=device),
            lstm_c=torch.zeros(1, policy.hidden_size, device=device),
        )

    frame_limit = args.get("save_frames")
    frames: list[Any] = []
    completed: list[dict[str, float]] = []

    try:
        ob, _ = env.reset(seed=int(args["train"].get("seed", 0)))
        while len(completed) < episodes:
            render = env.render() if render_enabled else None
            if frame_limit is not None and render is not None and len(frames) < frame_limit:
                frames.append(render)

            with torch.no_grad():
                ob_t = torch.as_tensor(ob).to(device)
                logits, value = policy.forward_eval(ob_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)

            ob, reward, terminal, truncated, info = env.step(int(action.reshape(-1)[0].item()))
            if not info:
                continue

            for item in info:
                if not item or float(item.get("n", 0.0)) <= 0.0:
                    continue
                completed.append({key: float(value) for key, value in item.items()})
                if len(completed) >= episodes:
                    break
    finally:
        env.close()

    if frames and len(frames) == frame_limit and args.get("gif_path") is not None:
        import imageio

        imageio.mimsave(args["gif_path"], frames, fps=args.get("fps") or 12.0, loop=0)
        print(f"Saved {len(frames)} frames to {args['gif_path']}")

    overall = summarize_episodes(completed)
    active_goal_rows = build_slice_rows(
        completed,
        "task_active_goals",
        lambda episode: int(round(float(episode.get("task_active_goals", 0.0)))),
    )
    task_family_rows = build_slice_rows(completed, "task_family", family_label)
    step_budget_rows = build_slice_rows(
        completed,
        "task_step_budget",
        lambda episode: step_budget_bucket(int(round(float(episode.get("task_step_budget", 0.0)))), step_bin_width),
    )

    report = {
        "episodes": completed,
        "overall": overall,
        "slices": {
            "task_active_goals": active_goal_rows,
            "task_family": task_family_rows,
            "task_step_budget": step_budget_rows,
        },
    }

    print("\nOverall")
    print(json.dumps(overall, indent=2, sort_keys=True))
    print_slice_rows("By Active Goals", active_goal_rows)
    print_slice_rows("By Task Family", task_family_rows)
    print_slice_rows("By Step Budget", step_budget_rows)
    return report


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
    eval_parser.add_argument("--episodes", type=int, default=128)
    eval_parser.add_argument("--slice-step-bin-width", type=int, default=4)
    eval_parser.add_argument("--json-out", default=None)
    eval_parser.add_argument("--save-frames", type=int, default=None)
    eval_parser.add_argument("--gif-path", default=None)
    eval_parser.add_argument("--fps", type=float, default=None)

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

    report = run_candy_crush_eval(args, episodes=cli.episodes, step_bin_width=cli.slice_step_bin_width)
    if cli.json_out is not None:
        json_path = Path(cli.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
