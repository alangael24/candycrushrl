#ifndef PUFFERLIB_FOOTBALL_HEAD_H
#define PUFFERLIB_FOOTBALL_HEAD_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define FH_OBS 14
#define FH_ACTIONS 4
#define FH_WIDTH 1080
#define FH_HEIGHT 720

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float r;
    int side;
} HeadPlayer;

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float r;
} HeadBall;

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float goals_scored;
    float goals_allowed;
    float wins;
    float draws;
    float n;
} Log;

typedef struct Client Client;
struct Client {
    int active;
};

typedef struct FootballHead FootballHead;
struct FootballHead {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    Client* client;

    int max_steps;
    int max_score;
    int tick;
    int player_score;
    int enemy_score;

    float gravity;
    float move_speed;
    float jump_velocity;
    float kick_velocity;
    float goal_reward;
    float touch_reward;
    float progress_reward;
    float alive_reward;

    HeadPlayer player;
    HeadPlayer enemy;
    HeadBall ball;
    float episode_return;
};

static const float FH_GROUND = 0.08f;
static const float FH_CEILING = 0.98f;
static const float FH_GOAL_HEIGHT = 0.34f;
static const float FH_WALL_BOUNCE = 0.88f;
static const float FH_FLOOR_BOUNCE = 0.82f;

static const Color FH_BG = (Color){134, 211, 255, 255};
static const Color FH_GRASS = (Color){74, 176, 88, 255};
static const Color FH_LINE = (Color){245, 245, 245, 255};
static const Color FH_GOAL = (Color){230, 230, 230, 255};
static const Color FH_BALL = (Color){255, 255, 255, 255};
static const Color FH_BALL_OUTLINE = (Color){40, 40, 40, 255};
static const Color FH_PLAYER = (Color){255, 188, 66, 255};
static const Color FH_ENEMY = (Color){82, 148, 255, 255};
static const Color FH_TEXT = (Color){32, 48, 64, 255};
static const Color FH_EYE = (Color){245, 245, 245, 255};

#ifndef KEY_A
#define KEY_A 65
#endif
#ifndef KEY_D
#define KEY_D 68
#endif
#ifndef KEY_W
#define KEY_W 87
#endif

static inline float fh_abs(float v) { return v < 0.0f ? -v : v; }

static inline float fh_clamp(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static inline float fh_randf(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

static inline bool fh_on_ground(HeadPlayer* player) {
    return player->y <= FH_GROUND + player->r + 0.0001f;
}

static inline void fh_reset_round(FootballHead* env) {
    env->player.x = 0.22f;
    env->player.y = FH_GROUND + env->player.r;
    env->player.vx = 0.0f;
    env->player.vy = 0.0f;

    env->enemy.x = 0.78f;
    env->enemy.y = FH_GROUND + env->enemy.r;
    env->enemy.vx = 0.0f;
    env->enemy.vy = 0.0f;

    env->ball.x = 0.50f + fh_randf(-0.03f, 0.03f);
    env->ball.y = 0.42f + fh_randf(-0.02f, 0.02f);
    env->ball.vx = fh_randf(-0.01f, 0.01f);
    env->ball.vy = 0.0f;
}

static inline void fh_init(FootballHead* env) {
    env->tick = 0;
    env->player_score = 0;
    env->enemy_score = 0;
    env->episode_return = 0.0f;
    env->client = NULL;
    memset(&env->log, 0, sizeof(Log));

    env->player.r = 0.065f;
    env->player.side = 1;
    env->enemy.r = 0.065f;
    env->enemy.side = -1;
    env->ball.r = 0.035f;
}

static inline void fh_reset(FootballHead* env) {
    env->tick = 0;
    env->player_score = 0;
    env->enemy_score = 0;
    env->episode_return = 0.0f;
    fh_reset_round(env);
}

static inline void fh_apply_player_input(HeadPlayer* player, int left, int right, int jump, float move_speed, float jump_velocity) {
    player->vx = 0.0f;
    if (left && !right) player->vx = -move_speed;
    if (right && !left) player->vx = move_speed;
    if (jump && fh_on_ground(player)) player->vy = jump_velocity;
}

static inline void fh_bot_policy(FootballHead* env, int* left, int* right, int* jump, int* kick) {
    const float desired_x = fh_clamp(env->ball.x + 0.06f, 0.55f, 0.92f);
    const float dx = desired_x - env->enemy.x;
    const float ball_dx = fh_abs(env->ball.x - env->enemy.x);

    *left = dx < -0.02f;
    *right = dx > 0.02f;
    *jump = fh_on_ground(&env->enemy) && env->ball.y > env->enemy.y + 0.08f && ball_dx < 0.18f;
    *kick = ball_dx < 0.14f && env->ball.y < env->enemy.y + 0.14f;
}

static inline void fh_integrate_player(HeadPlayer* player, float gravity) {
    player->vy += gravity;
    player->x += player->vx;
    player->y += player->vy;

    if (player->x - player->r < 0.02f) player->x = 0.02f + player->r;
    if (player->x + player->r > 0.98f) player->x = 0.98f - player->r;

    if (player->y - player->r < FH_GROUND) {
        player->y = FH_GROUND + player->r;
        if (player->vy < 0.0f) player->vy = 0.0f;
    }
    if (player->y + player->r > FH_CEILING) {
        player->y = FH_CEILING - player->r;
        if (player->vy > 0.0f) player->vy = 0.0f;
    }
}

static inline void fh_resolve_player_overlap(HeadPlayer* a, HeadPlayer* b) {
    const float dx = b->x - a->x;
    const float dy = b->y - a->y;
    const float dist2 = dx * dx + dy * dy;
    const float min_dist = a->r + b->r;
    if (dist2 <= 0.0f || dist2 >= min_dist * min_dist) return;

    const float dist = sqrtf(dist2);
    const float nx = dx / dist;
    const float ny = dy / dist;
    const float push = (min_dist - dist) * 0.5f;

    a->x -= nx * push;
    a->y -= ny * push;
    b->x += nx * push;
    b->y += ny * push;
}

static inline bool fh_ball_bounce(HeadBall* ball, HeadPlayer* player, float* reward, float touch_reward) {
    const float dx = ball->x - player->x;
    const float dy = ball->y - player->y;
    const float min_dist = ball->r + player->r;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 <= 0.0f || dist2 >= min_dist * min_dist) return false;

    const float dist = sqrtf(dist2);
    const float nx = dx / dist;
    const float ny = dy / dist;
    const float overlap = min_dist - dist;

    ball->x += nx * overlap;
    ball->y += ny * overlap;

    {
        const float rel_vx = ball->vx - player->vx;
        const float rel_vy = ball->vy - player->vy;
        const float impact = rel_vx * nx + rel_vy * ny;
        if (impact < 0.0f) {
            ball->vx -= 1.85f * impact * nx;
            ball->vy -= 1.85f * impact * ny;
        }
    }

    ball->vx += player->vx * 0.20f;
    ball->vy += player->vy * 0.15f;
    if (reward != NULL) *reward += touch_reward;
    return true;
}

static inline void fh_apply_kick(FootballHead* env, HeadPlayer* player, int kick, int side) {
    if (!kick) return;
    if (fh_abs(env->ball.x - player->x) > 0.14f) return;
    if (fh_abs(env->ball.y - player->y) > 0.16f) return;

    env->ball.vx += env->kick_velocity * (float)side;
    env->ball.vy += env->kick_velocity * 0.55f;
}

static inline int fh_goal_event(FootballHead* env) {
    if (env->ball.x - env->ball.r <= 0.0f && env->ball.y <= FH_GOAL_HEIGHT) return -1;
    if (env->ball.x + env->ball.r >= 1.0f && env->ball.y <= FH_GOAL_HEIGHT) return 1;
    return 0;
}

static inline void fh_integrate_ball(FootballHead* env, float* reward) {
    const float prev_x = env->ball.x;

    env->ball.vy += env->gravity;
    env->ball.x += env->ball.vx;
    env->ball.y += env->ball.vy;

    if (env->ball.y - env->ball.r < FH_GROUND) {
        env->ball.y = FH_GROUND + env->ball.r;
        if (env->ball.vy < 0.0f) env->ball.vy = -env->ball.vy * FH_FLOOR_BOUNCE;
        env->ball.vx *= 0.995f;
    }
    if (env->ball.y + env->ball.r > FH_CEILING) {
        env->ball.y = FH_CEILING - env->ball.r;
        if (env->ball.vy > 0.0f) env->ball.vy = -env->ball.vy * FH_WALL_BOUNCE;
    }

    if (env->ball.x - env->ball.r < 0.0f && env->ball.y > FH_GOAL_HEIGHT) {
        env->ball.x = env->ball.r;
        if (env->ball.vx < 0.0f) env->ball.vx = -env->ball.vx * FH_WALL_BOUNCE;
    }
    if (env->ball.x + env->ball.r > 1.0f && env->ball.y > FH_GOAL_HEIGHT) {
        env->ball.x = 1.0f - env->ball.r;
        if (env->ball.vx > 0.0f) env->ball.vx = -env->ball.vx * FH_WALL_BOUNCE;
    }

    env->ball.vx = fh_clamp(env->ball.vx, -0.06f, 0.06f);
    env->ball.vy = fh_clamp(env->ball.vy, -0.08f, 0.08f);
    *reward += env->progress_reward * (env->ball.x - prev_x);
}

static inline void fh_compute_observations(FootballHead* env) {
    env->observations[0] = env->player.x;
    env->observations[1] = env->player.y;
    env->observations[2] = env->player.vx / env->move_speed;
    env->observations[3] = env->player.vy / env->jump_velocity;
    env->observations[4] = env->ball.x - env->player.x;
    env->observations[5] = env->ball.y - env->player.y;
    env->observations[6] = env->ball.vx / env->kick_velocity;
    env->observations[7] = env->ball.vy / env->kick_velocity;
    env->observations[8] = env->enemy.x - env->player.x;
    env->observations[9] = env->enemy.y - env->player.y;
    env->observations[10] = env->enemy.vx / env->move_speed;
    env->observations[11] = env->enemy.vy / env->jump_velocity;
    env->observations[12] = (float)(env->player_score - env->enemy_score) / (float)env->max_score;
    env->observations[13] = (float)(env->max_steps - env->tick) / (float)env->max_steps;
}

static inline void fh_log_episode(FootballHead* env, int outcome) {
    env->log.score += env->player_score - env->enemy_score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.goals_scored += env->player_score;
    env->log.goals_allowed += env->enemy_score;
    env->log.wins += outcome > 0 ? 1.0f : 0.0f;
    env->log.draws += outcome == 0 ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static inline void c_reset(FootballHead* env) {
    fh_reset(env);
    fh_compute_observations(env);
}

static inline void c_step(FootballHead* env) {
    int enemy_left, enemy_right, enemy_jump, enemy_kick;
    float reward = env->alive_reward;
    int goal = 0;

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->tick += 1;

    fh_apply_player_input(
        &env->player,
        env->actions[0] > 0,
        env->actions[1] > 0,
        env->actions[2] > 0,
        env->move_speed,
        env->jump_velocity
    );

    fh_bot_policy(env, &enemy_left, &enemy_right, &enemy_jump, &enemy_kick);
    fh_apply_player_input(&env->enemy, enemy_left, enemy_right, enemy_jump, env->move_speed, env->jump_velocity);

    fh_integrate_player(&env->player, env->gravity);
    fh_integrate_player(&env->enemy, env->gravity);
    fh_resolve_player_overlap(&env->player, &env->enemy);

    fh_apply_kick(env, &env->player, env->actions[3] > 0, 1);
    fh_apply_kick(env, &env->enemy, enemy_kick, -1);
    fh_integrate_ball(env, &reward);
    fh_ball_bounce(&env->ball, &env->player, &reward, env->touch_reward);
    fh_ball_bounce(&env->ball, &env->enemy, NULL, 0.0f);

    goal = fh_goal_event(env);
    if (goal > 0) {
        env->player_score += 1;
        reward += env->goal_reward;
        fh_reset_round(env);
    } else if (goal < 0) {
        env->enemy_score += 1;
        reward -= env->goal_reward;
        fh_reset_round(env);
    }

    {
        const bool max_score_reached = env->player_score >= env->max_score || env->enemy_score >= env->max_score;
        const bool timeout = env->tick >= env->max_steps;
        if (timeout && env->player_score != env->enemy_score) {
            reward += env->player_score > env->enemy_score ? 0.5f * env->goal_reward : -0.5f * env->goal_reward;
        }

        env->episode_return += reward;
        env->rewards[0] = reward;

        if (max_score_reached || timeout) {
            const int outcome = env->player_score > env->enemy_score ? 1 : (env->player_score < env->enemy_score ? -1 : 0);
            env->terminals[0] = 1;
            fh_log_episode(env, outcome);
            fh_reset(env);
        }
    }

    fh_compute_observations(env);
}

static inline Client* fh_make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(FH_WIDTH, FH_HEIGHT, "PufferLib Football Head");
    SetTargetFPS(60);
    client->active = 1;
    return client;
}

static inline int fh_px(float x) { return (int)(x * FH_WIDTH); }
static inline int fh_py(float y) { return (int)((1.0f - y) * (FH_HEIGHT - 80)); }
static inline int fh_scale(float value) { return (int)(value * FH_WIDTH); }

static inline void fh_draw_player(HeadPlayer* player, Color color) {
    const int px = fh_px(player->x);
    const int py = fh_py(player->y);
    const int radius = fh_scale(player->r);
    DrawCircle(px, py, radius, color);
    DrawCircleLines(px, py, radius, FH_TEXT);
    DrawCircle(px - radius / 3, py - radius / 5, radius / 5, FH_EYE);
    DrawCircle(px + radius / 3, py - radius / 5, radius / 5, FH_EYE);
}

static inline void c_render(FootballHead* env) {
    char label[128];

    if (!IsWindowReady()) env->client = fh_make_client();
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(FH_BG);
    DrawRectangle(0, fh_py(FH_GROUND), FH_WIDTH, FH_HEIGHT, FH_GRASS);
    DrawRectangle(0, fh_py(FH_GROUND), FH_WIDTH, 3, FH_LINE);
    DrawRectangle(FH_WIDTH / 2 - 1, fh_py(FH_CEILING), 3, fh_py(FH_GROUND) - fh_py(FH_CEILING), FH_LINE);
    DrawRectangleLines(0, fh_py(FH_GOAL_HEIGHT), fh_scale(0.10f), fh_py(FH_GROUND) - fh_py(FH_GOAL_HEIGHT), FH_GOAL);
    DrawRectangleLines(FH_WIDTH - fh_scale(0.10f), fh_py(FH_GOAL_HEIGHT), fh_scale(0.10f), fh_py(FH_GROUND) - fh_py(FH_GOAL_HEIGHT), FH_GOAL);

    DrawCircle(fh_px(env->ball.x), fh_py(env->ball.y), fh_scale(env->ball.r), FH_BALL);
    DrawCircleLines(fh_px(env->ball.x), fh_py(env->ball.y), fh_scale(env->ball.r), FH_BALL_OUTLINE);
    fh_draw_player(&env->player, FH_PLAYER);
    fh_draw_player(&env->enemy, FH_ENEMY);

    snprintf(label, sizeof(label), "Score %d - %d", env->player_score, env->enemy_score);
    DrawText(label, 16, 16, 28, FH_TEXT);
    snprintf(label, sizeof(label), "Steps %d/%d", env->tick, env->max_steps);
    DrawText(label, 16, 48, 24, FH_TEXT);
    DrawText("A/D = move, W = jump, SPACE = kick", 16, 78, 22, FH_TEXT);
    EndDrawing();
}

static inline void c_close(FootballHead* env) {
    if (IsWindowReady()) CloseWindow();
    if (env->client != NULL) {
        free(env->client);
        env->client = NULL;
    }
}

#endif
