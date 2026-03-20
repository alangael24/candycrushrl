#include "flappy_bird.h"

int main() {
    FlappyBird env = {
        .max_steps = 1024,
        .gravity = -0.0035f,
        .flap_velocity = 0.032f,
        .pipe_speed = 0.014f,
        .pipe_spacing = 0.45f,
        .pipe_width = 0.12f,
        .gap_size = 0.30f,
        .bird_x = 0.28f,
        .bird_radius = 0.03f,
        .gap_margin = 0.10f,
        .alive_reward = 0.01f,
        .pass_reward = 1.0f,
        .death_penalty = -1.0f,
    };

    env.observations = (float*)calloc(FLAPPY_OBS, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyPressed(KEY_SPACE) ? 1 : 0;
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
