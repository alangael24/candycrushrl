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
