import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.football_head import binding


class FootballHead(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        max_steps=1800,
        max_score=3,
        gravity=-0.0025,
        move_speed=0.018,
        jump_velocity=0.045,
        kick_velocity=0.060,
        goal_reward=1.0,
        touch_reward=0.01,
        progress_reward=0.002,
        alive_reward=0.001,
        render_mode='human',
        log_interval=128,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FH_OBS,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.MultiDiscrete([2, 2, 2, 2])

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_steps=max_steps,
            max_score=max_score,
            gravity=gravity,
            move_speed=move_speed,
            jump_velocity=jump_velocity,
            kick_velocity=kick_velocity,
            goal_reward=goal_reward,
            touch_reward=touch_reward,
            progress_reward=progress_reward,
            alive_reward=alive_reward,
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


FH_OBS = 14


if __name__ == '__main__':
    env = FootballHead(num_envs=1024)
    env.reset()
    cache = 1024
    actions = np.random.randint(0, 2, size=(cache, env.num_agents, 4), dtype=np.int32)

    import time

    tick = 0
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[tick % cache])
        tick += 1

    print('FootballHead SPS:', int(env.num_agents * tick / (time.time() - start)))
