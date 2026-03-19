# PufferLib Candy Crush

This is a local Ocean-style environment implemented in C for throughput and wired
into the standard PufferLib CLI as `puffer_candy_crush`.

## Shape

- Single-agent Candy Crush style puzzle
- Action space: `board_size * board_size * 4`
- Observation space: flattened planes for candies/specials plus jelly,
  frosting, ingredients, goal progress, move budget, and legal-action mask

## Mechanics

- 4 in a line creates striped candies
- 5 in a line creates a color bomb
- T/L intersections create wrapped candies
- 2x2 squares create fish
- Direct striped, wrapped, color bomb, and fish combos resolve in C
- Level goals: `score`, `clear all jelly`, `drop all ingredients`,
  `collect a target color`, or `clear blocker tiles`
- Blockers: layered frosting plus jelly underlay tiles
- Ingredients: non-swappable pieces that fall with gravity and exit from the bottom row
- Legal action masking: the observation includes a per-action legality mask and
  the CandyCrush policy masks invalid swaps before sampling
- Campaign levels: the early level bank starts with "collect blue candies"
  stages, then transitions into blocker-clearing stages that unlock closed-off
  board regions, with pre-placed starter specials on selected levels
- Adaptive curriculum: unlocks harder authored levels when the frontier win
  rate crosses a threshold and still replays earlier levels occasionally

## CLI

Once the fork is installed or run from source, the environment is exposed as:

```bash
python -m pufferlib.pufferl train puffer_candy_crush
python -m pufferlib.pufferl eval puffer_candy_crush
```

Useful curriculum knobs in `candy_crush.ini` or CLI env overrides:

- `level_id = -1` keeps curriculum enabled; set `0..11` to lock a fixed level
- `curriculum_mode = 1` enables adaptive progression from easy to hard
- `curriculum_min_episodes`, `curriculum_threshold`, `curriculum_replay_prob`
  control unlock speed and replay of earlier levels
- `target_color`, `color_target`, and `frosting_target` define the explicit
  target when you disable curriculum and run a fixed custom level profile

## Notes

- This follows the same Ocean pattern as `template`, `snake`, and `g2048`.
- On the current upstream `setup.py`, Ocean C builds are supported on Linux/macOS.
- For Windows, use WSL or PufferTank-style Linux tooling for native builds.
