#include "block_puzzle.h"

#define OBS_SIZE OBS_TOTAL_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {ACTION_COUNT}
#define OBS_TENSOR_T ByteTensor
#define ACTION_MASK_OFFSET 178
#define ACTION_MASK_SIZE 1200

#define Env BlockPuzzle
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->board_size = (int)dict_get(kwargs, "board_size")->value;
    env->allow_rotations = (int)dict_get(kwargs, "allow_rotations")->value;
    env->reward_per_block = (float)dict_get(kwargs, "reward_per_block")->value;
    env->line_bonus = (float)dict_get(kwargs, "line_bonus")->value;
    env->multi_line_bonus = (float)dict_get(kwargs, "multi_line_bonus")->value;
    env->invalid_penalty = (float)dict_get(kwargs, "invalid_penalty")->value;
    env->loss_penalty = (float)dict_get(kwargs, "loss_penalty")->value;
    env->shaping_gamma = (float)dict_get(kwargs, "shaping_gamma")->value;
    env->legal_reward_scale = (float)dict_get(kwargs, "legal_reward_scale")->value;
    env->future_flex_reward_scale = (float)dict_get(kwargs, "future_flex_reward_scale")->value;
    env->fill_penalty_scale = (float)dict_get(kwargs, "fill_penalty_scale")->value;
    env->fill_penalty_threshold = (float)dict_get(kwargs, "fill_penalty_threshold")->value;
    init_env(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "lines_cleared", log->lines_cleared);
    dict_set(out, "pieces_placed", log->pieces_placed);
    dict_set(out, "invalid_actions", log->invalid_actions);
    dict_set(out, "board_fill", log->board_fill);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
