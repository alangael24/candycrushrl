# PufferLib Candy Crush

This is a local Ocean-style environment implemented in C for throughput and wired
into the standard PufferLib CLI as `puffer_candy_crush`.

## Shape

- Single-agent Candy Crush style puzzle
- Action space: `board_size * board_size * 4`
- Observation space: flattened planes for candies/specials plus jelly,
  frosting, ingredients, a goal vector, a remaining-goal vector, a normalized
  remaining-goal vector, move budget, and legal-action mask

## Mechanics

- 4 in a line creates striped candies
- 5 in a line creates a color bomb
- T/L intersections create wrapped candies
- 2x2 squares create fish
- Direct striped, wrapped, color bomb, and fish combos resolve in C
- Goal specification is vector-valued: the first `num_candies` slots track
  candy colors, followed by `jelly`, `frosting`, `ingredient`, and optional
  `score`
- Multiple goal slots can be non-zero at once, so compound tasks such as
  "collect blue and clear frosting" are representable directly
- Blockers: layered frosting plus jelly underlay tiles
- Ingredients: non-swappable pieces that fall with gravity and exit from the bottom row
- Legal action masking: the observation includes a per-action legality mask and
  the CandyCrush policy masks invalid swaps before sampling
- Campaign levels: the early level bank starts with "collect blue candies"
  stages, then transitions into blocker-clearing stages that unlock closed-off
  board regions, with pre-placed starter specials on selected levels
- Goal-conditioned reward shaping: dense reward uses PBRS on the normalized
  remaining-goal vector, with terminal win/loss terms and an efficiency bonus
  for winning early
- Adaptive curriculum: unlocks harder authored levels when the frontier win
  rate crosses a threshold and still replays earlier levels occasionally
- Task distribution mode: when enabled, resets sample compound goals and move
  budgets directly instead of relying on authored level profiles

## CLI

Once the fork is installed or run from source, the environment is exposed as:

```bash
python -m pufferlib.pufferl train puffer_candy_crush
python -m pufferlib.pufferl eval puffer_candy_crush
```

For serious runs, prefer the explicit presets instead of the ambiguous default
`candy_crush.ini`. The preset runner keeps the environment name stable while
loading a specific config file:

```bash
python scripts/candy_crush_train_preset.py train --preset a0-taskdist
python scripts/candy_crush_train_preset.py train --preset a0-campaign
python scripts/candy_crush_train_preset.py train --preset throughput
```

On the profiled 4090 host, the documented learning baseline uses the `A0`
training regime with `128 x 16` host geometry and `task_distribution_mode = 1`.
That preset is the recommended starting point for the first policy baseline:

```bash
python scripts/candy_crush_train_preset.py train \
  --preset a0-taskdist \
  --device cuda \
  --seed 101
```

`B0-taskdist` and `B0-campaign` are evaluation protocols, not new environment
IDs:

- `B0-taskdist`: run `a0-taskdist` for multiple seeds against a fixed eval split
- `B0-campaign`: run `a0-campaign` for multiple seeds to measure authored-level
  progression

The runner accepts `b0-taskdist` and `b0-campaign` as aliases to make that
intent explicit from the command line:

```bash
python scripts/candy_crush_train_preset.py train --preset b0-taskdist --seed 101
python scripts/candy_crush_train_preset.py train --preset b0-campaign --seed 101
```

Useful curriculum knobs in `candy_crush.ini` or CLI env overrides:

- `level_id = -1` keeps curriculum enabled; set `0..11` to lock a fixed level
- `curriculum_mode = 1` enables adaptive progression from easy to hard
- `curriculum_min_episodes`, `curriculum_threshold`, `curriculum_replay_prob`
  control unlock speed and replay of earlier levels
- `goal_vector` defines explicit custom targets when you disable curriculum and
  task distribution;
  for `num_candies=6`, a vector can be `[red, green, blue, yellow, purple,
  teal, jelly, frosting, ingredient, score]`
- `task_distribution_mode`, `task_min_active_goals`, `task_max_active_goals`,
  `task_min_steps`, and `task_max_steps` control sampling from `p(tau)`;
  task sampling treats each color slot independently, so compound goals like
  `[red, green, blue, yellow, purple, teal, jelly, frosting, ingredient, score]
  = [10, 0, 10, 0, 0, 0, 0, 20, 0, 0]` can be generated automatically
- `task_distribution_mode = 2` enables a mixed sampler that keeps the normal
  task distribution for most episodes while routing a configurable fraction
  through a harder bucket with more `3-goal`, `frosting`, and `ingredient`
  tasks
- `progress_reward_scale`, `shaping_gamma`, `success_bonus`,
  `failure_penalty`, and `efficiency_bonus` control the goal-aligned reward

## Presets

The repo now ships three explicit Candy Crush configs:

- `pufferlib/config/ocean/candy_crush_a0_taskdist.ini`
  `A0` learning regime, `task_distribution_mode = 1`, `curriculum_mode = 0`
- `pufferlib/config/ocean/candy_crush_a0_campaign.ini`
  `A0` learning regime, `task_distribution_mode = 0`, `curriculum_mode = 1`
- `pufferlib/config/ocean/candy_crush_mixed_hard_ft.ini`
  200M fine-tune preset with `task_distribution_mode = 2`, `65/35` normal/hard
  sampling, and the same `128 x 16` host geometry as the frozen screen protocol
- `pufferlib/config/ocean/candy_crush_throughput.ini`
  throughput-oriented stress test, not the preferred learning baseline

Override host geometry per machine from the preset runner when needed:

```bash
python scripts/candy_crush_train_preset.py train \
  --preset a0-taskdist \
  --device cuda \
  --env-num-envs 128 \
  --vec-num-envs 16
```

Fine-tune the best MLP checkpoint against the mixed sampler like this:

```bash
python scripts/candy_crush_train_preset.py train \
  --preset mixed-hard-ft \
  --device cuda \
  --seed 101 \
  --load-model-path /path/to/model.pt

python scripts/candy_crush_fixed_eval.py \
  --preset screen-200m \
  --device cpu \
  --load-model-path /path/to/fine_tuned_model.pt \
  --json-out artifacts/candy_crush_fixed_eval.json
```

## Guardrails

The `A0` learning guardrail now defaults to the explicit `a0-taskdist` config
instead of whatever happens to be in the default `candy_crush.ini`:

```bash
python scripts/candy_crush_a0_guardrail.py record artifacts/candy_crush_a0.json
python scripts/candy_crush_a0_guardrail.py check artifacts/candy_crush_a0.json
```

## Notes

- This follows the same Ocean pattern as `template`, `snake`, and `g2048`.
- On the current upstream `setup.py`, Ocean C builds are supported on Linux/macOS.
- For Windows, use WSL or PufferTank-style Linux tooling for native builds.
