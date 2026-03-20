#include "football_head.h"

int main() {
    FootballHead env = {
        .max_steps = 1800,
        .max_score = 3,
        .gravity = -0.0025f,
        .move_speed = 0.018f,
        .jump_velocity = 0.045f,
        .kick_velocity = 0.060f,
        .goal_reward = 1.0f,
        .touch_reward = 0.01f,
        .progress_reward = 0.002f,
        .alive_reward = 0.001f,
    };

    env.observations = (float*)calloc(FH_OBS, sizeof(float));
    env.actions = (int*)calloc(FH_ACTIONS, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    fh_init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyDown(KEY_A);
        env.actions[1] = IsKeyDown(KEY_D);
        env.actions[2] = IsKeyPressed(KEY_W);
        env.actions[3] = IsKeyDown(KEY_SPACE);
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
