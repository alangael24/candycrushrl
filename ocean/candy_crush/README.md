# Candy Crush

Native `4.0` port of the local Candy Crush style env. It keeps the old mechanics,
but uses the static Ocean layout:

- env name: `candy_crush`
- build: `bash build.sh candy_crush`
- run: `puffer train candy_crush`

This static build fixes the compile-time shape to:

- `board_size = 8`
- `num_candies = 6`
- action count `= 8 * 8 * 4 = 256`
- observation size `= 2464`

`vecenv` only passes numeric scalars through `Dict`, so the old `goal_vector`
list is exposed here as individual keys in `config/candy_crush.ini`:

- `goal_red`
- `goal_green`
- `goal_blue`
- `goal_yellow`
- `goal_purple`
- `goal_teal`
- `goal_jelly`
- `goal_frosting`
- `goal_ingredient`
- `goal_score`

The rest of the environment logic is the original authored/task-distribution
Candy Crush implementation with curriculum, blockers, ingredients, specials,
and legal-action masking in the observation.
