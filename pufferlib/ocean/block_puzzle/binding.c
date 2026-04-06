#include "block_puzzle.h"

#define Env BlockPuzzle
#define MY_SEED 1

static void my_seed(Env* env, unsigned int seed) {
    seed_rng(env, (uint64_t)seed);
}

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->board_size = unpack(kwargs, "board_size");
    env->allow_rotations = unpack(kwargs, "allow_rotations");
    env->reward_per_block = unpack(kwargs, "reward_per_block");
    env->line_bonus = unpack(kwargs, "line_bonus");
    env->multi_line_bonus = unpack(kwargs, "multi_line_bonus");
    env->invalid_penalty = unpack(kwargs, "invalid_penalty");
    env->loss_penalty = unpack(kwargs, "loss_penalty");
    init_env(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "lines_cleared", log->lines_cleared);
    assign_to_dict(dict, "pieces_placed", log->pieces_placed);
    assign_to_dict(dict, "invalid_actions", log->invalid_actions);
    assign_to_dict(dict, "board_fill", log->board_fill);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

