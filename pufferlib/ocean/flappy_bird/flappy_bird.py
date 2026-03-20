import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.flappy_bird import binding


class FlappyBird(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        max_steps=1024,
        gravity=-0.0035,
        flap_velocity=0.032,
        pipe_speed=0.014,
        pipe_spacing=0.45,
        pipe_width=0.12,
        gap_size=0.30,
        bird_x=0.28,
        bird_radius=0.03,
        gap_margin=0.10,
        alive_reward=0.01,
        pass_reward=1.0,
        death_penalty=-1.0,
        render_mode='human',
        report_interval=128,
        buf=None,
        seed=0,
    ):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(2)

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
            gravity=gravity,
            flap_velocity=flap_velocity,
            pipe_speed=pipe_speed,
            pipe_spacing=pipe_spacing,
            pipe_width=pipe_width,
            gap_size=gap_size,
            bird_x=bird_x,
            bird_radius=bird_radius,
            gap_margin=gap_margin,
            alive_reward=alive_reward,
            pass_reward=pass_reward,
            death_penalty=death_penalty,
        )

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
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


def test_performance(timeout=10, action_cache=8192):
    num_envs = 4096
    env = FlappyBird(num_envs=num_envs)
    env.reset()
    actions = np.random.randint(0, env.single_action_space.n, (action_cache, num_envs), dtype=np.int32)
    tick = 0

    import time

    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % action_cache])
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')


if __name__ == '__main__':
    test_performance()
