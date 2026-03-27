#include "candy_crush.h"

int main() {
    CandyCrush env = {
        .board_size = 8,
        .num_candies = 6,
        .max_steps = 40,
        .frosting_layers = 2,
        .ingredient_spawn_rows = 2,
        .task_distribution_mode = 1,
        .task_min_active_goals = 1,
        .task_max_active_goals = 3,
        .task_min_steps = 22,
        .task_max_steps = 40,
        .reward_per_tile = 0.05f,
        .combo_bonus = 0.10f,
        .invalid_penalty = -0.20f,
        .shuffle_penalty = 0.0f,
        .progress_reward_scale = 1.0f,
        .shaping_gamma = 0.995f,
        .success_bonus = 3.0f,
        .failure_penalty = 1.0f,
        .efficiency_bonus = 0.5f,
        .jelly_density = 0.35f,
        .frosting_density = 0.10f,
        .level_id = -1,
        .curriculum_mode = 0,
        .curriculum_start_level = 0,
        .curriculum_max_level = 11,
        .curriculum_min_episodes = 16,
        .curriculum_threshold = 0.40f,
        .curriculum_replay_prob = 0.15f,
    };

    const int obs_size = obs_feature_size(&env) + action_count(&env);
    env.observations = (unsigned char*)calloc(obs_size, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    init_env(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = rng_int_bounded(&env, env.board_size * env.board_size * 4);
            c_step(&env);
        }
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
