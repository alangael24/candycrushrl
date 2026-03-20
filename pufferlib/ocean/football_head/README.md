# PufferLib Football Head

Simple Head Soccer style Ocean environment:

- single-agent vs scripted opponent
- vector observation of player, ball, enemy, score, and time
- actions are `left`, `right`, `jump`, `kick`
- match ends on `max_score` or `max_steps`
- local play mode against a trained checkpoint

CLI:

```bash
python -m pufferlib.pufferl train puffer_football_head
python -m pufferlib.pufferl eval puffer_football_head
python -m pufferlib.ocean.football_head.play_vs_model --checkpoint /path/to/model.pt
```
