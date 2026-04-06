"""10x10 block puzzle environment with a native C backend."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.block_puzzle import binding


class BlockPuzzle(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        board_size=10,
        allow_rotations=1,
        reward_per_block=0.10,
        line_bonus=1.0,
        multi_line_bonus=0.5,
        invalid_penalty=-0.25,
        loss_penalty=-1.0,
        render_mode="human",
        log_interval=128,
        buf=None,
        seed=0,
    ):
        preview_cells = 3 * 5 * 5
        action_count = 3 * board_size * board_size * 4
        obs_size = board_size * board_size + preview_cells + 3 + action_count

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(action_count)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        self.board_size = board_size
        self.allow_rotations = allow_rotations
        self.reward_per_block = reward_per_block
        self.line_bonus = line_bonus
        self.multi_line_bonus = multi_line_bonus
        self.invalid_penalty = invalid_penalty
        self.loss_penalty = loss_penalty

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
            allow_rotations=self.allow_rotations,
            reward_per_block=self.reward_per_block,
            line_bonus=self.line_bonus,
            multi_line_bonus=self.multi_line_bonus,
            invalid_penalty=self.invalid_penalty,
            loss_penalty=self.loss_penalty,
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


if __name__ == "__main__":
    env = BlockPuzzle(num_envs=128)
    env.reset()

    cache = 1024
    actions = np.random.randint(0, env.single_action_space.n, size=(cache, env.num_agents))

    steps = 0
    import time

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % cache])
        steps += 1

    print("BlockPuzzle SPS:", int(env.num_agents * steps / (time.time() - start)))
    env.close()

