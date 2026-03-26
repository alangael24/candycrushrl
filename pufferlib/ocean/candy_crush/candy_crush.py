'''Single-agent Candy Crush style environment with a C backend.'''

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
        goal_vector=None,
        frosting_layers=2,
        ingredient_spawn_rows=2,
        task_distribution_mode=1,
        task_min_active_goals=1,
        task_max_active_goals=3,
        task_min_steps=22,
        task_max_steps=40,
        reward_per_tile=0.05,
        combo_bonus=0.10,
        invalid_penalty=-0.20,
        shuffle_penalty=0.0,
        jelly_reward=0.20,
        frosting_reward=0.10,
        ingredient_reward=1.0,
        color_reward=0.35,
        color_tile_scale=0.20,
        color_combo_scale=0.50,
        progress_reward_scale=1.0,
        success_bonus=3.0,
        failure_penalty=1.0,
        efficiency_bonus=0.5,
        jelly_density=0.35,
        frosting_density=0.10,
        level_id=-1,
        curriculum_mode=1,
        curriculum_start_level=0,
        curriculum_max_level=11,
        curriculum_min_episodes=32,
        curriculum_threshold=0.65,
        curriculum_replay_prob=0.15,
        render_mode='human',
        log_interval=128,
        buf=None,
        seed=0,
    ):
        goal_slots = num_candies + 4
        obs_size = (
            board_size * board_size * (num_candies * 5 + 4)
            + (goal_slots * 3 + 2)
            + board_size * board_size * 4
        )
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
        if goal_vector is not None:
            goal_vector = list(goal_vector)
            if len(goal_vector) > goal_slots:
                raise ValueError(f'goal_vector length must be <= {goal_slots} for num_candies={num_candies}')
        self.goal_vector = goal_vector
        self.frosting_layers = frosting_layers
        self.ingredient_spawn_rows = ingredient_spawn_rows
        self.task_distribution_mode = task_distribution_mode
        self.task_min_active_goals = task_min_active_goals
        self.task_max_active_goals = task_max_active_goals
        self.task_min_steps = task_min_steps
        self.task_max_steps = task_max_steps
        self.reward_per_tile = reward_per_tile
        self.combo_bonus = combo_bonus
        self.invalid_penalty = invalid_penalty
        self.shuffle_penalty = shuffle_penalty
        self.jelly_reward = jelly_reward
        self.frosting_reward = frosting_reward
        self.ingredient_reward = ingredient_reward
        self.color_reward = color_reward
        self.color_tile_scale = color_tile_scale
        self.color_combo_scale = color_combo_scale
        self.progress_reward_scale = progress_reward_scale
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.efficiency_bonus = efficiency_bonus
        self.jelly_density = jelly_density
        self.frosting_density = frosting_density
        self.level_id = level_id
        self.curriculum_mode = curriculum_mode
        self.curriculum_start_level = curriculum_start_level
        self.curriculum_max_level = curriculum_max_level
        self.curriculum_min_episodes = curriculum_min_episodes
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_replay_prob = curriculum_replay_prob
        if (
            self.goal_vector is None
            and self.task_distribution_mode == 0
            and self.level_id < 0
            and self.curriculum_mode == 0
        ):
            raise ValueError(
                'Provide goal_vector or enable task_distribution_mode/curriculum_mode/set level_id'
            )

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
            goal_vector=self.goal_vector,
            frosting_layers=self.frosting_layers,
            ingredient_spawn_rows=self.ingredient_spawn_rows,
            task_distribution_mode=self.task_distribution_mode,
            task_min_active_goals=self.task_min_active_goals,
            task_max_active_goals=self.task_max_active_goals,
            task_min_steps=self.task_min_steps,
            task_max_steps=self.task_max_steps,
            reward_per_tile=self.reward_per_tile,
            combo_bonus=self.combo_bonus,
            invalid_penalty=self.invalid_penalty,
            shuffle_penalty=self.shuffle_penalty,
            jelly_reward=self.jelly_reward,
            frosting_reward=self.frosting_reward,
            ingredient_reward=self.ingredient_reward,
            color_reward=self.color_reward,
            color_tile_scale=self.color_tile_scale,
            color_combo_scale=self.color_combo_scale,
            progress_reward_scale=self.progress_reward_scale,
            success_bonus=self.success_bonus,
            failure_penalty=self.failure_penalty,
            efficiency_bonus=self.efficiency_bonus,
            jelly_density=self.jelly_density,
            frosting_density=self.frosting_density,
            level_id=self.level_id,
            curriculum_mode=self.curriculum_mode,
            curriculum_start_level=self.curriculum_start_level,
            curriculum_max_level=self.curriculum_max_level,
            curriculum_min_episodes=self.curriculum_min_episodes,
            curriculum_threshold=self.curriculum_threshold,
            curriculum_replay_prob=self.curriculum_replay_prob,
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
