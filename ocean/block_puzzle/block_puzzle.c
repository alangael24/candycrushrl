#include <time.h>

#include "block_puzzle.h"

static int sample_legal_action(BlockPuzzle* env) {
    unsigned char* mask = env->observations + OBS_BOARD_CELLS + OBS_PREVIEW_CELLS + OBS_SCALAR_CELLS;
    int legal_count = 0;
    int action;

    for (action = 0; action < ACTION_COUNT; action++) {
        legal_count += mask[action] != 0;
    }

    if (legal_count == 0) {
        return 0;
    }

    {
        int choice = rng_int_bounded(env, legal_count);
        for (action = 0; action < ACTION_COUNT; action++) {
            if (!mask[action]) {
                continue;
            }
            if (choice == 0) {
                return action;
            }
            choice--;
        }
    }

    return 0;
}

int main(void) {
    BlockPuzzle env = {
        .num_agents = 1,
        .board_size = BOARD_SIZE,
        .allow_rotations = 1,
        .reward_per_block = 0.10f,
        .line_bonus = 1.0f,
        .multi_line_bonus = 0.5f,
        .invalid_penalty = -0.25f,
        .loss_penalty = -1.0f,
        .rng = (unsigned int)time(NULL),
    };

    allocate(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = (float)sample_legal_action(&env);
        c_step(&env);
        c_render(&env);
    }

    free_allocated(&env);
    return 0;
}
