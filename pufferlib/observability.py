from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import pufferlib


DEFAULT_OUTCOME_KEYS = (
    "episode_return",
    "episode_length",
    "score",
    "reward",
    "perf",
    "success",
    "failure",
    "win",
    "loss",
    "terminated",
    "truncated",
)

DEFAULT_EVENT_KEYS = (
    "event",
    "events",
    "death",
    "kill",
    "hit",
    "collision",
    "goal",
    "pickup",
    "spawn",
    "timeout",
    "success",
    "failure",
    "win",
    "loss",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix or "value": value}

    flat = {}
    for key, child in value.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flat.update(flatten_dict(child, next_prefix))
        else:
            flat[next_prefix] = child
    return flat


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_csv(text: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if text is None:
        return default

    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    return tuple(values) if values else default


def is_signal(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return value is not None


def select_matching(flat: dict[str, Any], keys: tuple[str, ...], signal_only: bool = False) -> dict[str, Any]:
    selected = {}
    for key, value in flat.items():
        lowered = key.lower()
        if not any(match in lowered for match in keys):
            continue
        if signal_only and not is_signal(value):
            continue
        selected[key] = to_jsonable(value)
    return selected


def find_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists():
            return path
    return None


def git_context(start: Path, include_status: bool = False, include_describe: bool = False) -> dict[str, Any]:
    repo_root = find_git_root(start)
    if repo_root is None:
        return {
            "root": None,
            "commit": None,
            "branch": None,
            "describe": None,
            "dirty": None,
            "status": [],
        }

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    describe = run_git("describe", "--always", "--dirty") if include_describe else None
    status = run_git("status", "--short") if include_status else None
    return {
        "root": str(repo_root),
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": describe,
        "dirty": ("-dirty" in describe) if describe is not None else (bool(status) if status is not None else None),
        "status": status.splitlines() if status else [],
    }


class AsyncFileWriter:
    def __init__(self, metrics_path: Path, episodes_path: Path, events_path: Path, queue_size: int) -> None:
        self.metrics_path = metrics_path
        self.episodes_path = episodes_path
        self.events_path = events_path
        self.queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=max(1, queue_size))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, channel: str, payload: dict[str, Any]) -> bool:
        try:
            self.queue.put_nowait((channel, payload))
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return

            channel, payload = item
            if channel == "metrics":
                path = self.metrics_path
            elif channel == "episodes":
                path = self.episodes_path
            else:
                path = self.events_path
            append_jsonl(path, payload)


class CompositeLogger:
    def __init__(self, *loggers: Any) -> None:
        self.loggers = [logger for logger in loggers if logger is not None]
        self.run_id = next(
            (logger.run_id for logger in self.loggers if hasattr(logger, "run_id")),
            str(int(100 * time.time())),
        )

    def log(self, logs: dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(logs, step)

    def close(self, model_path: str, early_stop: bool) -> None:
        for logger in self.loggers:
            logger.close(model_path, early_stop)

    def on_run_start(self, **kwargs: Any) -> None:
        for logger in self.loggers:
            callback = getattr(logger, "on_run_start", None)
            if callable(callback):
                callback(**kwargs)

    def on_env_infos(self, infos: list[dict[str, Any]], *, step: int, uptime: float) -> None:
        for logger in self.loggers:
            callback = getattr(logger, "on_env_infos", None)
            if callable(callback):
                callback(infos, step=step, uptime=uptime)


class SessionLogger:
    def __init__(self, args: dict[str, Any], env_name: str, run_id: str | None = None) -> None:
        self.args = args
        self.env_name = env_name
        self.started_at = time.time()
        self.run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

        root = Path(args["observe_root"]).resolve()
        self.root = ensure_dir(root)
        self.runs_dir = ensure_dir(root / "runs")
        self.run_dir = ensure_dir(self.runs_dir / self.run_id)
        self.manifest_path = self.run_dir / "manifest.json"
        self.summary_path = self.run_dir / "summary.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.events_path = self.run_dir / "events.jsonl"

        self.writer = AsyncFileWriter(
            metrics_path=self.metrics_path,
            episodes_path=self.episodes_path,
            events_path=self.events_path,
            queue_size=args["observe_queue_size"],
        )

        self.metric_rows = 0
        self.episode_rows = 0
        self.key_event_rows = 0
        self.event_rows = 0
        self.dropped_metrics = 0
        self.dropped_episodes = 0
        self.dropped_key_events = 0
        self.dropped_events = 0
        self.include_code_version = bool(args.get("observe_code_version", True))
        self.include_model_sha256 = bool(args.get("observe_model_sha256", False))
        self.include_git_status = bool(args.get("observe_git_status", False))
        self.include_git_describe = bool(args.get("observe_git_describe", False))
        self.outcome_keys = parse_csv(args.get("observe_outcome_keys"), DEFAULT_OUTCOME_KEYS)
        self.event_keys = parse_csv(args.get("observe_event_keys"), DEFAULT_EVENT_KEYS)
        self.keep_env_reports = bool(args.get("observe_env_reports", False))

    def on_run_start(self, *, args: dict[str, Any], env_name: str, policy: Any, vecenv: Any) -> None:
        policy_class = f"{policy.__class__.__module__}.{policy.__class__.__name__}"
        configured_version = args.get("observe_model_version")
        model_version = policy_class if configured_version in (None, "", "auto") else configured_version

        checkpoint_source = args.get("load_model_path")
        checkpoint_sha = None
        if self.include_model_sha256 and checkpoint_source not in (None, "latest"):
            checkpoint_sha = sha256_file(checkpoint_source)
        note = args.get("observe_note")
        code_version = None
        if self.include_code_version:
            code_version = git_context(
                Path(__file__).resolve(),
                include_status=self.include_git_status,
                include_describe=self.include_git_describe,
            )

        manifest = {
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "env_name": env_name,
            "pufferlib_version": pufferlib.__version__,
            "seeds": {
                "train_seed": args["train"].get("seed"),
                "vec_seed": args["vec"].get("seed"),
            },
            "config": to_jsonable(args),
            "model": {
                "version": model_version,
                "policy_class": policy_class,
                "parameter_count": int(sum(p.numel() for p in policy.parameters())),
                "load_model_path": checkpoint_source,
                "load_id": args.get("load_id"),
                "checkpoint_sha256": checkpoint_sha,
            },
            "vectorization": {
                "backend": args["vec"].get("backend"),
                "num_envs": args["vec"].get("num_envs"),
                "num_agents": getattr(vecenv, "num_agents", None),
                "agents_per_batch": getattr(vecenv, "agents_per_batch", None),
            },
            "code_version": code_version,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "tag": args.get("tag"),
            "note": note,
        }
        write_json(self.manifest_path, manifest)

    def log(self, logs: dict[str, Any], step: int) -> None:
        payload = {
            "event": "aggregate_metrics",
            "run_id": self.run_id,
            "recorded_at": utc_now_iso(),
            "agent_steps": step,
            "metrics": to_jsonable(logs),
        }
        if self.writer.put("metrics", payload):
            self.metric_rows += 1
        else:
            self.dropped_metrics += 1

    def on_env_infos(self, infos: list[dict[str, Any]], *, step: int, uptime: float) -> None:
        for info in infos:
            if not isinstance(info, dict) or not info:
                continue

            flat = flatten_dict(info)
            outcomes = select_matching(flat, self.outcome_keys)
            key_events = select_matching(flat, self.event_keys, signal_only=True)
            recorded_at = utc_now_iso()

            if outcomes:
                payload = {
                    "event": "episode_summary",
                    "run_id": self.run_id,
                    "recorded_at": recorded_at,
                    "agent_steps": step,
                    "uptime": round(uptime, 6),
                    "outcomes": outcomes,
                }
                if self.writer.put("episodes", payload):
                    self.episode_rows += 1
                else:
                    self.dropped_episodes += 1

            if key_events:
                payload = {
                    "event": "key_event",
                    "run_id": self.run_id,
                    "recorded_at": recorded_at,
                    "agent_steps": step,
                    "uptime": round(uptime, 6),
                    "key_events": key_events,
                }
                if self.writer.put("events", payload):
                    self.key_event_rows += 1
                else:
                    self.dropped_key_events += 1

            if not self.keep_env_reports:
                continue

            payload = {
                "event": "env_info",
                "run_id": self.run_id,
                "recorded_at": recorded_at,
                "agent_steps": step,
                "uptime": round(uptime, 6),
                "info": to_jsonable(info),
            }
            if outcomes:
                payload["outcomes"] = outcomes
            if key_events:
                payload["key_events"] = key_events

            if self.writer.put("events", payload):
                self.event_rows += 1
            else:
                self.dropped_events += 1

    def close(self, model_path: str, early_stop: bool) -> None:
        self.writer.close()
        model_sha = sha256_file(model_path) if self.include_model_sha256 else None
        summary = {
            "run_id": self.run_id,
            "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "ended_at": utc_now_iso(),
            "duration_s": round(time.time() - self.started_at, 6),
            "metrics_written": self.metric_rows,
            "episode_summaries_written": self.episode_rows,
            "key_events_written": self.key_event_rows,
            "env_reports_written": self.event_rows,
            "dropped_metrics": self.dropped_metrics,
            "dropped_episode_summaries": self.dropped_episodes,
            "dropped_key_events": self.dropped_key_events,
            "dropped_env_reports": self.dropped_events,
            "early_stop": early_stop,
            "model_artifact": {
                "path": model_path,
                "sha256": model_sha,
            },
        }
        write_json(self.summary_path, summary)
