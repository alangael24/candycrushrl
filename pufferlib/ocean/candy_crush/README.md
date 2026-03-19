# PufferLib Candy Crush

This is a local Ocean-style environment implemented in C for throughput and wired
into the standard PufferLib CLI as `puffer_candy_crush`.

## Shape

- Single-agent Candy Crush style puzzle
- Action space: `board_size * board_size * 4`
- Observation space: flattened planes for candies/specials plus jelly,
  frosting, ingredients, goal progress, and move budget

## Mechanics

- 4 in a line creates striped candies
- 5 in a line creates a color bomb
- T/L intersections create wrapped candies
- 2x2 squares create fish
- Direct striped, wrapped, color bomb, and fish combos resolve in C
- Level goals: `score`, `clear all jelly`, or `drop all ingredients`
- Blockers: layered frosting plus jelly underlay tiles
- Ingredients: non-swappable pieces that fall with gravity and exit from the bottom row

## CLI

Once the fork is installed or run from source, the environment is exposed as:

```bash
python -m pufferlib.pufferl train puffer_candy_crush
python -m pufferlib.pufferl eval puffer_candy_crush
```

## Notes

- This follows the same Ocean pattern as `template`, `snake`, and `g2048`.
- On the current upstream `setup.py`, Ocean C builds are supported on Linux/macOS.
- For Windows, use WSL or PufferTank-style Linux tooling for native builds.
