#ifndef PUFFERLIB_FLAPPY_BIRD_H
#define PUFFERLIB_FLAPPY_BIRD_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define FLAPPY_OBS 8
#define FLAPPY_PIPES 3
#define FLAPPY_WIDTH 960
#define FLAPPY_HEIGHT 540

typedef struct {
    float x;
    float gap_y;
    int passed;
} Pipe;

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float pipes_passed;
    float collisions;
    float timeouts;
    float n;
} Log;

typedef struct Client Client;
struct Client {
    int active;
};

typedef struct FlappyBird FlappyBird;
struct FlappyBird {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    Client* client;

    int max_steps;
    int tick;
    int score;

    float gravity;
    float flap_velocity;
    float pipe_speed;
    float pipe_spacing;
    float pipe_width;
    float gap_size;
    float bird_x;
    float bird_y;
    float bird_vel;
    float bird_radius;
    float gap_margin;
    float alive_reward;
    float pass_reward;
    float death_penalty;
    float episode_return;

    Pipe pipes[FLAPPY_PIPES];
};

static const Color FLAPPY_BG = (Color){131, 211, 247, 255};
static const Color FLAPPY_GROUND = (Color){215, 196, 117, 255};
static const Color FLAPPY_PIPE = (Color){102, 198, 59, 255};
static const Color FLAPPY_PIPE_DARK = (Color){71, 145, 41, 255};
static const Color FLAPPY_BIRD = (Color){255, 224, 82, 255};
static const Color FLAPPY_TEXT = (Color){32, 48, 64, 255};
static const Color FLAPPY_WHITE = (Color){245, 245, 245, 255};

static inline float flappy_randf(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

static inline float flappy_clampf(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static inline float flappy_sample_gap(FlappyBird* env) {
    const float half_gap = env->gap_size * 0.5f;
    const float low = env->gap_margin + half_gap;
    const float high = 1.0f - env->gap_margin - half_gap;
    return flappy_randf(low, high);
}

static inline float flappy_rightmost_pipe_x(FlappyBird* env) {
    float rightmost = env->pipes[0].x;
    for (int i = 1; i < FLAPPY_PIPES; i++) {
        if (env->pipes[i].x > rightmost) rightmost = env->pipes[i].x;
    }
    return rightmost;
}

static inline void flappy_spawn_pipe(FlappyBird* env, Pipe* pipe, float x) {
    pipe->x = x;
    pipe->gap_y = flappy_sample_gap(env);
    pipe->passed = 0;
}

static inline void flappy_reset_pipes(FlappyBird* env) {
    const float start_x = 0.80f;
    for (int i = 0; i < FLAPPY_PIPES; i++) {
        flappy_spawn_pipe(env, &env->pipes[i], start_x + i * env->pipe_spacing);
    }
}

static inline void flappy_next_pipes(FlappyBird* env, Pipe** next, Pipe** second) {
    Pipe* first = NULL;
    Pipe* second_local = NULL;
    for (int i = 0; i < FLAPPY_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        if (pipe->x + env->pipe_width * 0.5f < env->bird_x) continue;
        if (first == NULL || pipe->x < first->x) {
            second_local = first;
            first = pipe;
        } else if (second_local == NULL || pipe->x < second_local->x) {
            second_local = pipe;
        }
    }

    if (first == NULL) {
        first = &env->pipes[0];
        for (int i = 1; i < FLAPPY_PIPES; i++) {
            if (env->pipes[i].x < first->x) first = &env->pipes[i];
        }
    }

    if (second_local == NULL) {
        second_local = first;
        for (int i = 0; i < FLAPPY_PIPES; i++) {
            Pipe* pipe = &env->pipes[i];
            if (pipe == first) continue;
            if (second_local == first || pipe->x < second_local->x) second_local = pipe;
        }
    }

    *next = first;
    *second = second_local;
}

static inline void flappy_compute_observations(FlappyBird* env) {
    Pipe* next = NULL;
    Pipe* second = NULL;
    flappy_next_pipes(env, &next, &second);

    env->observations[0] = env->bird_y;
    env->observations[1] = env->bird_vel;
    env->observations[2] = next->x - env->bird_x;
    env->observations[3] = next->gap_y - env->bird_y;
    env->observations[4] = second->x - env->bird_x;
    env->observations[5] = second->gap_y - env->bird_y;
    env->observations[6] = env->gap_size;
    env->observations[7] = (float)(env->max_steps - env->tick) / (float)env->max_steps;
}

static inline bool flappy_pipe_collision(FlappyBird* env, Pipe* pipe) {
    const float left = pipe->x - env->pipe_width * 0.5f;
    const float right = pipe->x + env->pipe_width * 0.5f;
    const float bird_left = env->bird_x - env->bird_radius;
    const float bird_right = env->bird_x + env->bird_radius;
    const float gap_top = pipe->gap_y - env->gap_size * 0.5f;
    const float gap_bottom = pipe->gap_y + env->gap_size * 0.5f;
    const float bird_top = env->bird_y - env->bird_radius;
    const float bird_bottom = env->bird_y + env->bird_radius;

    if (bird_right < left || bird_left > right) return false;
    return bird_top < gap_top || bird_bottom > gap_bottom;
}

static inline bool flappy_collision(FlappyBird* env) {
    if (env->bird_y - env->bird_radius <= 0.0f) return true;
    if (env->bird_y + env->bird_radius >= 1.0f) return true;
    for (int i = 0; i < FLAPPY_PIPES; i++) {
        if (flappy_pipe_collision(env, &env->pipes[i])) return true;
    }
    return false;
}

static inline void flappy_log_episode(FlappyBird* env, bool collision, bool timeout) {
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.pipes_passed += env->score;
    env->log.collisions += collision ? 1.0f : 0.0f;
    env->log.timeouts += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static inline void init(FlappyBird* env) {
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0.0f;
    env->client = NULL;
    memset(&env->log, 0, sizeof(Log));
}

static inline void c_reset(FlappyBird* env) {
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0.0f;
    env->bird_y = flappy_randf(0.40f, 0.60f);
    env->bird_vel = 0.0f;
    flappy_reset_pipes(env);
    flappy_compute_observations(env);
}

static inline void flappy_update_world(FlappyBird* env) {
    env->bird_vel += env->gravity;
    env->bird_vel = flappy_clampf(env->bird_vel, -0.08f, 0.08f);
    env->bird_y += env->bird_vel;

    for (int i = 0; i < FLAPPY_PIPES; i++) {
        env->pipes[i].x -= env->pipe_speed;
    }

    for (int i = 0; i < FLAPPY_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        if (!pipe->passed && pipe->x + env->pipe_width * 0.5f < env->bird_x) {
            pipe->passed = 1;
            env->score += 1;
            env->rewards[0] += env->pass_reward;
        }
    }

    for (int i = 0; i < FLAPPY_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        if (pipe->x + env->pipe_width * 0.5f < -0.10f) {
            flappy_spawn_pipe(env, pipe, flappy_rightmost_pipe_x(env) + env->pipe_spacing);
        }
    }
}

static inline void c_step(FlappyBird* env) {
    const int action = env->actions[0];
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->tick += 1;

    if (action == 1) {
        env->bird_vel = env->flap_velocity;
    }

    flappy_update_world(env);
    env->rewards[0] += env->alive_reward;

    {
        const bool collision = flappy_collision(env);
        const bool timeout = env->tick >= env->max_steps;
        if (collision) env->rewards[0] += env->death_penalty;
        env->episode_return += env->rewards[0];

        if (collision || timeout) {
            env->terminals[0] = 1;
            flappy_log_episode(env, collision, timeout);
            c_reset(env);
            return;
        }
    }

    flappy_compute_observations(env);
}

static inline Client* flappy_make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(FLAPPY_WIDTH, FLAPPY_HEIGHT, "PufferLib Flappy Bird");
    SetTargetFPS(60);
    client->active = 1;
    return client;
}

static inline void c_render(FlappyBird* env) {
    const int ground_y = FLAPPY_HEIGHT - 80;
    const int bird_px = (int)(env->bird_x * FLAPPY_WIDTH);
    const int bird_py = (int)((1.0f - env->bird_y) * (ground_y - 20));
    char label[96];

    if (!IsWindowReady()) {
        env->client = flappy_make_client();
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(FLAPPY_BG);
    DrawRectangle(0, ground_y, FLAPPY_WIDTH, FLAPPY_HEIGHT - ground_y, FLAPPY_GROUND);

    for (int i = 0; i < FLAPPY_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        const int pipe_x = (int)((pipe->x - env->pipe_width * 0.5f) * FLAPPY_WIDTH);
        const int pipe_w = (int)(env->pipe_width * FLAPPY_WIDTH);
        const int gap_top = (int)((1.0f - (pipe->gap_y + env->gap_size * 0.5f)) * (ground_y - 20));
        const int gap_bottom = (int)((1.0f - (pipe->gap_y - env->gap_size * 0.5f)) * (ground_y - 20));
        DrawRectangle(pipe_x, 0, pipe_w, gap_top, FLAPPY_PIPE);
        DrawRectangle(pipe_x, gap_bottom, pipe_w, ground_y - gap_bottom, FLAPPY_PIPE);
        DrawRectangleLines(pipe_x, 0, pipe_w, gap_top, FLAPPY_PIPE_DARK);
        DrawRectangleLines(pipe_x, gap_bottom, pipe_w, ground_y - gap_bottom, FLAPPY_PIPE_DARK);
    }

    DrawCircle(bird_px, bird_py, (int)(env->bird_radius * FLAPPY_WIDTH), FLAPPY_BIRD);
    DrawCircleLines(bird_px, bird_py, (int)(env->bird_radius * FLAPPY_WIDTH), FLAPPY_TEXT);

    snprintf(label, sizeof(label), "Score: %d", env->score);
    DrawText(label, 16, 16, 24, FLAPPY_TEXT);
    snprintf(label, sizeof(label), "Steps: %d/%d", env->tick, env->max_steps);
    DrawText(label, 16, 46, 24, FLAPPY_TEXT);
    DrawText("SPACE = flap", 16, 76, 22, FLAPPY_WHITE);
    EndDrawing();
}

static inline void c_close(FlappyBird* env) {
    if (IsWindowReady()) CloseWindow();
    if (env->client != NULL) {
        free(env->client);
        env->client = NULL;
    }
}

#endif
