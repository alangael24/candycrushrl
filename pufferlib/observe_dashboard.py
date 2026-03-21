from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_metric(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        return None

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    metric_rows = load_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    episode_rows = load_jsonl(run_dir / "episodes.jsonl") if (run_dir / "episodes.jsonl").exists() else []

    last_metrics = metric_rows[-1]["metrics"] if metric_rows else {}
    episode_returns = []
    episode_lengths = []
    for row in episode_rows:
        outcomes = row.get("outcomes", {})
        ret = pick_metric(outcomes, "episode_return", "reward", "score")
        length = pick_metric(outcomes, "episode_length")
        if isinstance(ret, (int, float)):
            episode_returns.append(float(ret))
        if isinstance(length, (int, float)):
            episode_lengths.append(float(length))

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "env_name": manifest.get("env_name"),
        "created_at": manifest.get("created_at"),
        "model_version": manifest.get("model", {}).get("version"),
        "device": manifest.get("config", {}).get("train", {}).get("device"),
        "vec_envs": manifest.get("config", {}).get("vec", {}).get("num_envs"),
        "train_seed": manifest.get("seeds", {}).get("train_seed"),
        "duration_s": summary.get("duration_s"),
        "metrics_written": summary.get("metrics_written", 0),
        "episode_summaries_written": summary.get("episode_summaries_written", 0),
        "key_events_written": summary.get("key_events_written", 0),
        "last_sps": pick_metric(last_metrics, "SPS", "performance/SPS"),
        "last_agent_steps": pick_metric(last_metrics, "agent_steps"),
        "last_reward": pick_metric(last_metrics, "environment/reward", "reward"),
        "last_loss": pick_metric(last_metrics, "losses/value_loss", "value_loss"),
        "reward_rolling_mean": pick_metric(last_metrics, "observe/reward_rolling_mean"),
        "failure_rolling_rate": pick_metric(last_metrics, "observe/failure_rolling_rate"),
        "entropy_rolling": pick_metric(last_metrics, "observe/entropy_rolling"),
        "kl_rolling": pick_metric(last_metrics, "observe/kl_rolling"),
        "short_long_episode_ratio": pick_metric(last_metrics, "observe/short_long_episode_ratio"),
        "stagnation_rate": pick_metric(last_metrics, "observe/stagnation_rate"),
        "mean_episode_return": round(fmean(episode_returns), 4) if episode_returns else None,
        "mean_episode_length": round(fmean(episode_lengths), 2) if episode_lengths else None,
        "run_dir": str(run_dir),
    }


def render_dashboard(root: Path, runs: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('run_id', '')))}</td>"
            f"<td>{html.escape(str(run.get('env_name', '')))}</td>"
            f"<td>{html.escape(str(run.get('device', '')))}</td>"
            f"<td>{html.escape(str(run.get('vec_envs', '')))}</td>"
            f"<td>{html.escape(str(run.get('train_seed', '')))}</td>"
            f"<td>{html.escape(str(run.get('duration_s', '')))}</td>"
            f"<td>{html.escape(str(run.get('last_sps', '')))}</td>"
            f"<td>{html.escape(str(run.get('last_reward', '')))}</td>"
            f"<td>{html.escape(str(run.get('reward_rolling_mean', '')))}</td>"
            f"<td>{html.escape(str(run.get('failure_rolling_rate', '')))}</td>"
            f"<td>{html.escape(str(run.get('entropy_rolling', '')))}</td>"
            f"<td>{html.escape(str(run.get('kl_rolling', '')))}</td>"
            f"<td>{html.escape(str(run.get('short_long_episode_ratio', '')))}</td>"
            f"<td>{html.escape(str(run.get('stagnation_rate', '')))}</td>"
            f"<td>{html.escape(str(run.get('mean_episode_return', '')))}</td>"
            f"<td>{html.escape(str(run.get('mean_episode_length', '')))}</td>"
            f"<td>{html.escape(str(run.get('metrics_written', '')))}</td>"
            f"<td>{html.escape(str(run.get('episode_summaries_written', '')))}</td>"
            f"<td>{html.escape(str(run.get('key_events_written', '')))}</td>"
            f"<td><code>{html.escape(str(run.get('run_dir', '')))}</code></td>"
            "</tr>"
        )

    body = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Puffer Observe Dashboard</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f252b;
      --muted: #697682;
      --accent: #0f766e;
      --border: #d8d0c0;
    }
    body {
      margin: 0;
      padding: 32px;
      background:
        radial-gradient(circle at top left, #fef3c7 0, rgba(254,243,199,0) 35%),
        radial-gradient(circle at top right, #dbeafe 0, rgba(219,234,254,0) 30%),
        var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Iosevka Aile", "IBM Plex Sans", sans-serif;
    }
    h1 {
      margin: 0 0 8px;
      font: 700 28px/1.1 "IBM Plex Sans", sans-serif;
    }
    p {
      margin: 0 0 20px;
      color: var(--muted);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(31, 37, 43, 0.06);
      overflow: hidden;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #f7faf9;
      color: var(--accent);
      font-weight: 700;
    }
    code {
      font: 12px/1.4 "Iosevka Term", monospace;
      word-break: break-all;
    }
    .meta {
      display: flex;
      gap: 12px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: #ecfdf5;
      border: 1px solid #bbf7d0;
      color: #166534;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <h1>Puffer Observe Dashboard</h1>
  <p>Static summary for lightweight observe runs. Rebuild after new sessions.</p>
  <div class="meta">
    <span class="pill">root: __ROOT__</span>
    <span class="pill">runs: __RUNS__</span>
  </div>
  <div class="panel">
    <table>
      <thead>
        <tr>
          <th>run_id</th>
          <th>env</th>
          <th>device</th>
          <th>vec_envs</th>
          <th>seed</th>
          <th>duration_s</th>
          <th>last_sps</th>
          <th>last_reward</th>
          <th>reward_roll</th>
          <th>failure_roll</th>
          <th>entropy_roll</th>
          <th>kl_roll</th>
          <th>short_long</th>
          <th>stagnation</th>
          <th>mean_ep_return</th>
          <th>mean_ep_len</th>
          <th>metrics</th>
          <th>episodes</th>
          <th>events</th>
          <th>run_dir</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    rendered = (
        body.replace("__ROOT__", html.escape(str(root)))
        .replace("__RUNS__", str(len(runs)))
        .replace("__ROWS__", "\n".join(rows) if rows else '<tr><td colspan="20">No runs found.</td></tr>')
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static dashboard from observe run artifacts.")
    parser.add_argument("--root", type=Path, default=Path("observability"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    runs_dir = root / "runs"
    output = args.output.resolve() if args.output else root / "dashboard.html"

    runs = []
    if runs_dir.exists():
        for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), reverse=True):
            row = summarize_run(run_dir)
            if row is not None:
                runs.append(row)

    render_dashboard(root, runs, output)
    print(f"runs={len(runs)}")
    print(f"dashboard={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
