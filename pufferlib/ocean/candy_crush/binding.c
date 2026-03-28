#include "candy_crush.h"

#define Env CandyCrush
#define MY_SEED 1

static void my_seed(Env* env, unsigned int seed) {
    seed_rng(env, (uint64_t)seed);
}

#include "../env_binding.h"

static int unpack_goal_vector(PyObject* kwargs, const char* key, Env* env) {
    PyObject* value = PyDict_GetItemString(kwargs, key);
    const int slots = goal_slot_count(env);
    clear_goal_vector(env->goal_target);
    env->has_goal_vector = 0;

    if (value == NULL || value == Py_None) return 0;

    PyObject* seq = PySequence_Fast(value, "goal_vector must be a sequence of ints");
    if (seq == NULL) return -1;

    Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
    if (length < 1 || length > slots) {
        PyErr_Format(
            PyExc_ValueError,
            "goal_vector length must be in [1, %d] for num_candies=%d",
            slots,
            env->num_candies
        );
        Py_DECREF(seq);
        return -1;
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "goal_vector items must be ints");
            Py_DECREF(seq);
            return -1;
        }
        env->goal_target[i] = (int)PyLong_AsLong(item);
    }

    env->has_goal_vector = 1;
    Py_DECREF(seq);
    return 0;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->board_size = unpack(kwargs, "board_size");
    env->num_candies = unpack(kwargs, "num_candies");
    env->max_steps = unpack(kwargs, "max_steps");
    env->frosting_layers = unpack(kwargs, "frosting_layers");
    env->ingredient_spawn_rows = unpack(kwargs, "ingredient_spawn_rows");
    env->task_distribution_mode = unpack(kwargs, "task_distribution_mode");
    env->task_min_active_goals = unpack(kwargs, "task_min_active_goals");
    env->task_max_active_goals = unpack(kwargs, "task_max_active_goals");
    env->task_min_steps = unpack(kwargs, "task_min_steps");
    env->task_max_steps = unpack(kwargs, "task_max_steps");
    env->task_family_sampling_mode = unpack(kwargs, "task_family_sampling_mode");
    env->task_min_blocker_goals = unpack(kwargs, "task_min_blocker_goals");
    env->task_color_weight = unpack(kwargs, "task_color_weight");
    env->task_jelly_weight = unpack(kwargs, "task_jelly_weight");
    env->task_frosting_weight = unpack(kwargs, "task_frosting_weight");
    env->task_ingredient_weight = unpack(kwargs, "task_ingredient_weight");
    env->task_score_weight = unpack(kwargs, "task_score_weight");
    env->reward_per_tile = unpack(kwargs, "reward_per_tile");
    env->combo_bonus = unpack(kwargs, "combo_bonus");
    env->invalid_penalty = unpack(kwargs, "invalid_penalty");
    env->shuffle_penalty = unpack(kwargs, "shuffle_penalty");
    env->progress_reward_scale = unpack(kwargs, "progress_reward_scale");
    env->shaping_gamma = unpack(kwargs, "shaping_gamma");
    env->success_bonus = unpack(kwargs, "success_bonus");
    env->failure_penalty = unpack(kwargs, "failure_penalty");
    env->efficiency_bonus = unpack(kwargs, "efficiency_bonus");
    env->jelly_density = unpack(kwargs, "jelly_density");
    env->frosting_density = unpack(kwargs, "frosting_density");
    env->level_id = unpack(kwargs, "level_id");
    env->curriculum_mode = unpack(kwargs, "curriculum_mode");
    env->curriculum_start_level = unpack(kwargs, "curriculum_start_level");
    env->curriculum_max_level = unpack(kwargs, "curriculum_max_level");
    env->curriculum_min_episodes = unpack(kwargs, "curriculum_min_episodes");
    env->curriculum_threshold = unpack(kwargs, "curriculum_threshold");
    env->curriculum_replay_prob = unpack(kwargs, "curriculum_replay_prob");
    if (unpack_goal_vector(kwargs, "goal_vector", env) != 0) return -1;
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
    assign_to_dict(dict, "jelly_cleared", log->jelly_cleared);
    assign_to_dict(dict, "frosting_cleared", log->frosting_cleared);
    assign_to_dict(dict, "ingredient_dropped", log->ingredient_dropped);
    assign_to_dict(dict, "color_collected", log->color_collected);
    assign_to_dict(dict, "goal_progress", log->goal_progress);
    assign_to_dict(dict, "level_wins", log->level_wins);
    assign_to_dict(dict, "level_id", log->level_id);
    assign_to_dict(dict, "unlocked_level", log->unlocked_level);
    assign_to_dict(dict, "curriculum_win_rate", log->curriculum_win_rate);
    assign_to_dict(dict, "task_active_goals", log->task_active_goals);
    assign_to_dict(dict, "task_step_budget", log->task_step_budget);
    assign_to_dict(dict, "task_family_color", log->task_family_color);
    assign_to_dict(dict, "task_family_jelly", log->task_family_jelly);
    assign_to_dict(dict, "task_family_frosting", log->task_family_frosting);
    assign_to_dict(dict, "task_family_ingredient", log->task_family_ingredient);
    assign_to_dict(dict, "task_family_score", log->task_family_score);
    return 0;
}
