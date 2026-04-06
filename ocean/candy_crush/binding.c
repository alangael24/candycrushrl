#include "candy_crush.h"

#define OBS_SIZE CANDY_OBS_TOTAL_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CANDY_ACTION_COUNT}
#define OBS_TENSOR_T ByteTensor

#define Env CandyCrush
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->board_size = (int)dict_get(kwargs, "board_size")->value;
    env->num_candies = (int)dict_get(kwargs, "num_candies")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->frosting_layers = (int)dict_get(kwargs, "frosting_layers")->value;
    env->ingredient_spawn_rows = (int)dict_get(kwargs, "ingredient_spawn_rows")->value;
    env->task_distribution_mode = (int)dict_get(kwargs, "task_distribution_mode")->value;
    env->task_min_active_goals = (int)dict_get(kwargs, "task_min_active_goals")->value;
    env->task_max_active_goals = (int)dict_get(kwargs, "task_max_active_goals")->value;
    env->task_min_steps = (int)dict_get(kwargs, "task_min_steps")->value;
    env->task_max_steps = (int)dict_get(kwargs, "task_max_steps")->value;
    env->reward_per_tile = (float)dict_get(kwargs, "reward_per_tile")->value;
    env->combo_bonus = (float)dict_get(kwargs, "combo_bonus")->value;
    env->invalid_penalty = (float)dict_get(kwargs, "invalid_penalty")->value;
    env->shuffle_penalty = (float)dict_get(kwargs, "shuffle_penalty")->value;
    env->progress_reward_scale = (float)dict_get(kwargs, "progress_reward_scale")->value;
    env->shaping_gamma = (float)dict_get(kwargs, "shaping_gamma")->value;
    env->success_bonus = (float)dict_get(kwargs, "success_bonus")->value;
    env->failure_penalty = (float)dict_get(kwargs, "failure_penalty")->value;
    env->efficiency_bonus = (float)dict_get(kwargs, "efficiency_bonus")->value;
    env->jelly_density = (float)dict_get(kwargs, "jelly_density")->value;
    env->frosting_density = (float)dict_get(kwargs, "frosting_density")->value;
    env->level_id = (int)dict_get(kwargs, "level_id")->value;
    env->curriculum_mode = (int)dict_get(kwargs, "curriculum_mode")->value;
    env->curriculum_start_level = (int)dict_get(kwargs, "curriculum_start_level")->value;
    env->curriculum_max_level = (int)dict_get(kwargs, "curriculum_max_level")->value;
    env->curriculum_min_episodes = (int)dict_get(kwargs, "curriculum_min_episodes")->value;
    env->curriculum_threshold = (float)dict_get(kwargs, "curriculum_threshold")->value;
    env->curriculum_replay_prob = (float)dict_get(kwargs, "curriculum_replay_prob")->value;

    clear_goal_vector(env->goal_target);
    env->goal_target[0] = (int)dict_get(kwargs, "goal_red")->value;
    env->goal_target[1] = (int)dict_get(kwargs, "goal_green")->value;
    env->goal_target[2] = (int)dict_get(kwargs, "goal_blue")->value;
    env->goal_target[3] = (int)dict_get(kwargs, "goal_yellow")->value;
    env->goal_target[4] = (int)dict_get(kwargs, "goal_purple")->value;
    env->goal_target[5] = (int)dict_get(kwargs, "goal_teal")->value;
    env->goal_target[6] = (int)dict_get(kwargs, "goal_jelly")->value;
    env->goal_target[7] = (int)dict_get(kwargs, "goal_frosting")->value;
    env->goal_target[8] = (int)dict_get(kwargs, "goal_ingredient")->value;
    env->goal_target[9] = (int)dict_get(kwargs, "goal_score")->value;
    env->has_goal_vector = 0;
    env->rng_seed = 0;
    env->rng_state = 0;

    init_env(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "total_cleared", log->total_cleared);
    dict_set(out, "invalid_swaps", log->invalid_swaps);
    dict_set(out, "successful_swaps", log->successful_swaps);
    dict_set(out, "total_cascades", log->total_cascades);
    dict_set(out, "max_combo", log->max_combo);
    dict_set(out, "reshuffles", log->reshuffles);
    dict_set(out, "jelly_cleared", log->jelly_cleared);
    dict_set(out, "frosting_cleared", log->frosting_cleared);
    dict_set(out, "ingredient_dropped", log->ingredient_dropped);
    dict_set(out, "color_collected", log->color_collected);
    dict_set(out, "goal_progress", log->goal_progress);
    dict_set(out, "level_wins", log->level_wins);
    dict_set(out, "level_id", log->level_id);
    dict_set(out, "unlocked_level", log->unlocked_level);
    dict_set(out, "curriculum_win_rate", log->curriculum_win_rate);
    dict_set(out, "n", log->n);
}
