#include "football_head.h"

#define Env FootballHead
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_steps = unpack(kwargs, "max_steps");
    env->max_score = unpack(kwargs, "max_score");
    env->gravity = unpack(kwargs, "gravity");
    env->move_speed = unpack(kwargs, "move_speed");
    env->jump_velocity = unpack(kwargs, "jump_velocity");
    env->kick_velocity = unpack(kwargs, "kick_velocity");
    env->goal_reward = unpack(kwargs, "goal_reward");
    env->touch_reward = unpack(kwargs, "touch_reward");
    env->progress_reward = unpack(kwargs, "progress_reward");
    env->alive_reward = unpack(kwargs, "alive_reward");
    env->manual_enemy = unpack(kwargs, "manual_enemy");
    fh_init(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "goals_scored", log->goals_scored);
    assign_to_dict(dict, "goals_allowed", log->goals_allowed);
    assign_to_dict(dict, "wins", log->wins);
    assign_to_dict(dict, "draws", log->draws);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
