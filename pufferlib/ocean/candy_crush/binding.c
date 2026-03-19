#include "candy_crush.h"

#define Env CandyCrush
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->board_size = unpack(kwargs, "board_size");
    env->num_candies = unpack(kwargs, "num_candies");
    env->max_steps = unpack(kwargs, "max_steps");
    env->reward_per_tile = unpack(kwargs, "reward_per_tile");
    env->combo_bonus = unpack(kwargs, "combo_bonus");
    env->invalid_penalty = unpack(kwargs, "invalid_penalty");
    env->shuffle_penalty = unpack(kwargs, "shuffle_penalty");
    init_env(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "total_cleared", log->total_cleared);
    assign_to_dict(dict, "invalid_swaps", log->invalid_swaps);
    assign_to_dict(dict, "successful_swaps", log->successful_swaps);
    assign_to_dict(dict, "total_cascades", log->total_cascades);
    assign_to_dict(dict, "max_combo", log->max_combo);
    assign_to_dict(dict, "reshuffles", log->reshuffles);
    return 0;
}

