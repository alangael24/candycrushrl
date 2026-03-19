#include "candy_crush.h"

int main() {
    CandyCrush env = {
        .board_size = 8,
        .num_candies = 6,
        .max_steps = 40,
        .reward_per_tile = 0.05f,
        .combo_bonus = 0.10f,
        .invalid_penalty = -0.20f,
        .shuffle_penalty = 0.0f,
    };

    const int obs_size = env.board_size * env.board_size * (env.num_candies + 1);
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

