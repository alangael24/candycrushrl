# PufferLib 4.0 Block Puzzle

Native `4.0` Ocean environment for a single-agent `10x10` block puzzle with `3`
visible pieces.

State:
- board occupancy
- 3 visible piece previews
- 3 active-slot flags
- legal-action mask appended to the observation

Action:
- discrete `(piece_slot, row, col, rotation)` encoded into `1200` actions

Transition:
1. place the selected piece
2. clear full rows and columns
3. update score and reward
4. draw a new hand after all 3 pieces are consumed

Done:
- no visible piece can be legally placed

Reward:
- reward per placed block
- bonus per cleared line
- extra multi-line bonus
- terminal loss penalty

Build:
```bash
./build.sh block_puzzle
```

Train:
```bash
puffer train block_puzzle
python -m pufferlib.pufferl train block_puzzle
```
