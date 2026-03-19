#include "candy_crush.h"

int main() {
    CandyCrush env = {
        .board_size = 8,
        .num_candies = 6,
        .max_steps = 40,
        .objective_mode = 1,
        .score_target = 150,
        .frosting_layers = 2,
        .ingredient_target = 2,
        .ingredient_spawn_rows = 2,
        .target_color = 3,
        .color_target = 30,
        .frosting_target = 30,
        .reward_per_tile = 0.05f,
        .combo_bonus = 0.10f,
        .invalid_penalty = -0.20f,
        .shuffle_penalty = 0.0f,
        .jelly_reward = 0.20f,
        .frosting_reward = 0.10f,
        .ingredient_reward = 1.0f,
        .color_reward = 0.35f,
        .color_tile_scale = 0.20f,
        .color_combo_scale = 0.50f,
        .success_bonus = 3.0f,
        .jelly_density = 0.35f,
        .frosting_density = 0.10f,
        .level_id = -1,
        .curriculum_mode = 1,
        .curriculum_start_level = 0,
        .curriculum_max_level = 11,
        .curriculum_min_episodes = 16,
        .curriculum_threshold = 0.40f,
        .curriculum_replay_prob = 0.15f,
    };

    const int obs_size = env.board_size * env.board_size * (env.num_candies * 5 + 8) + env.board_size * env.board_size * 4;
    env.observations = (unsigned char*)calloc(obs_size, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    init_env(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = rand() % (env.board_size * env.board_size * 4);
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
