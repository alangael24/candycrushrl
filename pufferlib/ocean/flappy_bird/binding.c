#include "flappy_bird.h"

#define Env FlappyBird
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_steps = unpack(kwargs, "max_steps");
    env->gravity = unpack(kwargs, "gravity");
    env->flap_velocity = unpack(kwargs, "flap_velocity");
    env->pipe_speed = unpack(kwargs, "pipe_speed");
    env->pipe_spacing = unpack(kwargs, "pipe_spacing");
    env->pipe_width = unpack(kwargs, "pipe_width");
    env->gap_size = unpack(kwargs, "gap_size");
    env->bird_x = unpack(kwargs, "bird_x");
    env->bird_radius = unpack(kwargs, "bird_radius");
    env->gap_margin = unpack(kwargs, "gap_margin");
    env->alive_reward = unpack(kwargs, "alive_reward");
    env->pass_reward = unpack(kwargs, "pass_reward");
    env->death_penalty = unpack(kwargs, "death_penalty");
    init(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "pipes_passed", log->pipes_passed);
    assign_to_dict(dict, "collisions", log->collisions);
    assign_to_dict(dict, "timeouts", log->timeouts);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
