# PufferLib Candy Crush

This is a local Ocean-style environment implemented in C for throughput and wired
into the standard PufferLib CLI as `puffer_candy_crush`.

## Shape

- Single-agent match-3 puzzle
- Action space: `board_size * board_size * 4`
- Observation space: flattened one-hot candy planes plus one move-budget plane

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

