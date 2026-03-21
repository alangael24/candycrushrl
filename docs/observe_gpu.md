# Observe GPU Quick Start

This repo exposes a lightweight `observe` mode:

- captures config, model version, seeds, aggregated metrics
- captures cheap episode summaries by default
- avoids raw `env_info` payloads unless you opt in
- writes asynchronously under `--observe-root`

## Fresh clone

```bash
git clone https://github.com/alangael24/candycrushrl.git
cd candycrushrl
```

## Ubuntu / WSL setup

Use your existing WSL Python environment or create one, then install the project in editable mode. If you only need a subset of native envs, build just the trainer plus the envs you will test.

Example:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If you need to rebuild native envs in place:

```bash
python setup.py build_ext --inplace
python setup.py build_football_head
python setup.py build_candy_crush
```

## GPU smoke test on football head

Baseline:

```bash
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' \
python -m pufferlib.pufferl train puffer_football_head \
  --train.device cuda \
  --train.optimizer adam \
  --vec.num-envs 4 \
  --train.total-timesteps 8192 \
  --train.batch-size 4096 \
  --train.minibatch-size 4096 \
  --train.max-minibatch-size 4096 \
  --train.checkpoint-interval 999999
```

Observed:

```bash
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' \
python -m pufferlib.pufferl observe puffer_football_head \
  --train.device cuda \
  --train.optimizer adam \
  --vec.num-envs 4 \
  --train.total-timesteps 8192 \
  --train.batch-size 4096 \
  --train.minibatch-size 4096 \
  --train.max-minibatch-size 4096 \
  --train.checkpoint-interval 999999 \
  --observe-root ./observability_football_gpu \
  --observe-model-version football-head-gpu-v1
```

Default `observe` is intentionally lightweight. It does **not** record raw env reports unless you opt in.

## Build the dashboard

```bash
python -m pufferlib.observe_dashboard --root ./observability_football_gpu
```

This writes:

```text
observability_football_gpu/
  dashboard.html
  runs/<run_id>/
    manifest.json
    summary.json
    metrics.jsonl
    episodes.jsonl
    events.jsonl   # only when key events or raw env reports are recorded
```

## Heavier capture

If you explicitly want raw env reports:

```bash
python -m pufferlib.pufferl observe puffer_football_head \
  --train.device cuda \
  --observe-root ./observability_football_gpu_full \
  --observe-env-reports
```

That is useful for debugging, but it is not the default because it can add noticeable overhead.
