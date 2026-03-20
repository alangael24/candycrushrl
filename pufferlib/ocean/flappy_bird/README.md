# PufferLib Flappy Bird

Simple Flappy Bird style Ocean environment with:

- vector observations
- discrete actions: `0 = no flap`, `1 = flap`
- dense reward via survival reward plus pipe-pass reward
- terminal on pipe/ceiling/floor collision or `max_steps`

Observation layout:

1. bird y
2. bird velocity
3. next pipe horizontal distance
4. next pipe gap center relative to bird
5. second pipe horizontal distance
6. second pipe gap center relative to bird
7. gap size
8. remaining steps ratio

CLI:

```bash
python -m pufferlib.pufferl train puffer_flappy_bird
python -m pufferlib.pufferl eval puffer_flappy_bird
```
