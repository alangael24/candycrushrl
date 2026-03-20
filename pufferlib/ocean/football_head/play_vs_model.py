import argparse
from pathlib import Path

import numpy as np
import torch

from pufferlib.models import Default as Policy
from pufferlib.ocean.football_head.football_head import FootballHead


def parse_args():
    parser = argparse.ArgumentParser(description='Play Football Head against a trained checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to a .pt checkpoint')
    return parser.parse_args()


def greedy_multidiscrete_action(logits):
    if isinstance(logits, tuple):
        return torch.stack([head.argmax(dim=1) for head in logits], dim=1)
    return logits.argmax(dim=1, keepdim=True)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    env = FootballHead(
        num_envs=1,
        manual_enemy=True,
        render_mode='human',
        log_interval=1_000_000,
    )
    policy = Policy(env)
    state_dict = torch.load(checkpoint, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()

    print('Agent plays left side. You play right side.')
    print('Controls: LEFT/RIGHT move, UP jump, ENTER kick, ESC quit.')

    observations, _ = env.reset()
    env.render()
    try:
        while True:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
                logits, _ = policy.forward_eval(obs_tensor)
                actions = greedy_multidiscrete_action(logits).cpu().numpy().astype(np.int32)

            observations, _, terminals, truncations, _ = env.step(actions)
            env.render()
    finally:
        env.close()


if __name__ == '__main__':
    main()
