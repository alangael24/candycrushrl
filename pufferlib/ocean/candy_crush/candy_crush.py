'''Single-agent Candy Crush style match-3 environment with a C backend.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.candy_crush import binding


class CandyCrush(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        board_size=8,
        num_candies=6,
        max_steps=40,
        reward_per_tile=0.05,
        combo_bonus=0.10,
        invalid_penalty=-0.20,
        shuffle_penalty=0.0,
        render_mode='human',
        log_interval=128,
        buf=None,
        seed=0,
    ):
        obs_size = board_size * board_size * (num_candies + 1)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(board_size * board_size * 4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        self.board_size = board_size
        self.num_candies = num_candies
        self.max_steps = max_steps
        self.reward_per_tile = reward_per_tile
        self.combo_bonus = combo_bonus
        self.invalid_penalty = invalid_penalty
        self.shuffle_penalty = shuffle_penalty

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            board_size=self.board_size,
            num_candies=self.num_candies,
            max_steps=self.max_steps,
            reward_per_tile=self.reward_per_tile,
            combo_bonus=self.combo_bonus,
            invalid_penalty=self.invalid_penalty,
            shuffle_penalty=self.shuffle_penalty,
        )
        self.tick = 0

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    env = CandyCrush(num_envs=128)
    env.reset()

    cache = 1024
    actions = np.random.randint(0, env.single_action_space.n, size=(cache, env.num_agents))

    steps = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % cache])
        steps += 1

    print('CandyCrush SPS:', int(env.num_agents * steps / (time.time() - start)))
    env.close()

