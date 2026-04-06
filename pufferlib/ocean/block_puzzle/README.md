# PufferLib Block Puzzle

Single-agent 10x10 block puzzle with 3 visible pieces, modeled as a native
Ocean environment.

## State

- 10x10 board occupancy
- 3 currently visible pieces
- 3 active-slot flags
- legal-action mask

## Action

Discrete action decoded as `(piece_slot, row, col, rotation)` where:

- `piece_slot` is `0..2`
- `row`, `col` are the top-left anchor for the rotated piece footprint
- `rotation` is `0..3`

Invalid actions are masked in the observation. If sampled anyway, they incur a
small penalty and the board is unchanged.

## Transition

1. Place the selected visible piece.
2. Clear any completed rows and columns.
3. Score the placement and line clears.
4. When all 3 pieces are consumed, sample a fresh hand of 3 pieces.

## Done

- No visible piece can be legally placed anywhere on the board.

## Reward

- positive reward per block placed
- bonus per cleared line
- extra bonus for multi-line clears
- loss penalty when the episode ends

## CLI

```bash
python -m pufferlib.pufferl train puffer_block_puzzle
python -m pufferlib.pufferl eval puffer_block_puzzle
```

