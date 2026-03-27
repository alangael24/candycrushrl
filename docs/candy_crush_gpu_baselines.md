# Candy Crush GPU Baselines

This note records the first stable GPU training baselines for `puffer_candy_crush`
from a rented Linux GPU machine.

## Learning Baseline

This was the first configuration that clearly improved training metrics beyond
"it runs fast":

```bash
python -m pufferlib.pufferl train puffer_candy_crush \
  --train.device cuda \
  --env.num-envs 256 \
  --vec.num-envs 8 \
  --train.total-timesteps 10000000 \
  --train.minibatch-size 65536 \
  --train.max-minibatch-size 65536 \
  --train.update-epochs 4 \
  --train.learning-rate 0.01 \
  --train.optimizer adam
```

Observed direction:

- lower SPS than pure throughput runs
- much higher `explained_variance`
- higher `episode_return`
- higher `level_wins`
- shorter `episode_length`

## Throughput Baseline

This configuration maximized throughput during early scaling tests:

```bash
python -m pufferlib.pufferl train puffer_candy_crush \
  --train.device cuda \
  --env.num-envs 512 \
  --vec.num-envs 16 \
  --train.total-timesteps 10000000 \
  --train.minibatch-size 524288 \
  --train.max-minibatch-size 524288
```

Observed direction:

- about `747k SPS`
- good for stress-testing the pipeline
- not the best learning-oriented configuration so far

## Interpretation

For this environment, the best early throughput setup and the best early
learning setup are not the same.

- Use the learning baseline when comparing reward shaping, task sampling, or
  policy changes.
- Use the throughput baseline when checking systems regressions or env-side
  performance changes.

## Stage Closure

This phase is considered closed and should be treated as the reference point
for future Candy Crush environment work.

### Semantic Baseline

The semantic baseline is `A0`: compare learning changes against the stable PPO
training regime rather than against raw SPS.

```bash
python -m pufferlib.pufferl train puffer_candy_crush \
  --train.device cuda \
  --env.num-envs 256 \
  --vec.num-envs 8 \
  --train.total-timesteps 10000000 \
  --train.minibatch-size 65536 \
  --train.max-minibatch-size 65536 \
  --train.update-epochs 4 \
  --train.learning-rate 0.01 \
  --train.optimizer adam
```

### System Baseline

The system baseline is `A0` plus the env-side performance patches that were
validated without changing observable behavior:

- remove redundant `update_observations()` on the win path
- fuse legal-move detection with `action_mask` generation
- split `update_observations()` into board, meta, and mask slices
- replace the global observation `memset` with slice-local clearing for board
  and mask only
- cache swappable cells and reuse swap-legality inputs inside `action_mask`
  generation

Host geometry may still be tuned per machine as long as the learning regime is
kept fixed. On one rented 4090 host, `128 x 16` outperformed `256 x 8` while
preserving the same training regime.

### Performance Milestone

`main@58ba60d` is the current Candy Crush system baseline:

- semantic behavior preserved by step-by-step equivalence
- env-side observation and action-mask hot paths optimized without changing the
  observable MDP
- `A0` remains the learning baseline
- `128 x 16` is the documented host geometry for the tested 4090 rental host

Treat this as the default reference point when evaluating future performance
work on Candy Crush.

### Next Focus

The first environment hot-path phase is considered closed. Do not reopen
`write_board_obs`, `meta_obs`, dirty rectangles, incremental observations, or
mask zero-removal unless a new profile clearly makes them hot again.

The next systems-performance focus should be:

- `Copy` path and host-to-GPU movement
- host-specific `env x vec` geometry sweeps with `A0` fixed
- `vec_step` only if profiling still points at env CPU after `Copy` work

### Required Guardrail

Any future env-side performance change must pass step-by-step equivalence
before it is accepted.

```bash
mkdir -p artifacts

python scripts/candy_crush_equivalence.py record artifacts/candy_crush_trace.npz \
  --steps 4096 \
  --seed 1234 \
  --action-seed 2026

python scripts/candy_crush_equivalence.py check artifacts/candy_crush_trace.npz
```

Treat this as mandatory whenever changing simulation, observation encoding, or
action masking. Compare learning changes by timestep against `A0`; compare
systems changes by SPS against the system baseline.
