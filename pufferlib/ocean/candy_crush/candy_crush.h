#ifndef PUFFERLIB_CANDY_CRUSH_H
#define PUFFERLIB_CANDY_CRUSH_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define MAX_BOARD 10
#define MAX_CANDIES 8
#define MAX_COMPONENTS (MAX_BOARD * MAX_BOARD)
#define MAX_EFFECTS 1024
#define MAX_LEVEL_BANK 12
#define COLOR_MASK 0x0F
#define TYPE_SHIFT 4
#define SPECIAL_LAYERS 5

typedef enum { SPECIAL_NONE, SPECIAL_STRIPED_H, SPECIAL_STRIPED_V, SPECIAL_WRAPPED, SPECIAL_COLOR_BOMB, SPECIAL_FISH, SPECIAL_INGREDIENT } SpecialType;
typedef enum { EFFECT_ROW, EFFECT_COL, EFFECT_BLAST3, EFFECT_BLAST5, EFFECT_CROSS3, EFFECT_COLOR, EFFECT_BOARD, EFFECT_FISH, EFFECT_FISH_H, EFFECT_FISH_V, EFFECT_FISH_W } EffectKind;
typedef enum { GOAL_SCORE, GOAL_JELLY, GOAL_INGREDIENT, GOAL_COLOR, GOAL_FROSTING } GoalMode;

typedef struct {
    float score, episode_return, episode_length, total_cleared, invalid_swaps;
    float successful_swaps, total_cascades, max_combo, reshuffles;
    float jelly_cleared, frosting_cleared, ingredient_dropped, color_collected, goal_progress, level_wins, n;
    float level_id, unlocked_level, curriculum_win_rate;
} Log;

typedef struct { int kind, row, col, color, count; } Effect;
typedef struct { Effect items[MAX_EFFECTS]; int count; } EffectQueue;

typedef struct {
    bool matched[MAX_BOARD][MAX_BOARD];
    bool square[MAX_BOARD][MAX_BOARD];
    int h[MAX_BOARD][MAX_BOARD];
    int v[MAX_BOARD][MAX_BOARD];
    int component[MAX_BOARD][MAX_BOARD];
    int component_color[MAX_COMPONENTS];
    int components;
} MatchAnalysis;

typedef struct { int candies, jelly, frosting, ingredients, color; } ClearStats;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int board_size;
    int num_candies;
    int max_steps;
    int objective_mode;
    int score_target;
    int frosting_layers;
    int ingredient_target;
    int ingredient_spawn_rows;
    int target_color;
    int color_target;
    int frosting_target;

    float reward_per_tile;
    float combo_bonus;
    float invalid_penalty;
    float shuffle_penalty;
    float jelly_reward;
    float frosting_reward;
    float ingredient_reward;
    float success_bonus;
    float jelly_density;
    float frosting_density;

    int base_max_steps;
    int base_objective_mode;
    int base_score_target;
    int base_frosting_layers;
    int base_ingredient_target;
    int base_ingredient_spawn_rows;
    int base_target_color;
    int base_color_target;
    int base_frosting_target;
    float base_jelly_density;
    float base_frosting_density;

    int level_id;
    int curriculum_mode;
    int curriculum_start_level;
    int curriculum_max_level;
    int curriculum_min_episodes;
    int active_level;
    int unlocked_level;
    int frontier_level;
    int frontier_episodes;
    int frontier_wins;
    float curriculum_threshold;
    float curriculum_replay_prob;

    int steps, score, total_cleared, invalid_swaps, successful_swaps;
    int total_cascades, max_combo, reshuffles;
    int jelly_total, jelly_remaining, jelly_cleared, frosting_cleared;
    int ingredient_total, ingredient_remaining, ingredients_dropped, level_won;
    int color_collected;
    int starter_striped, starter_wrapped, starter_color_bomb, starter_fish;
    float episode_return;

    unsigned char board[MAX_BOARD][MAX_BOARD];
    unsigned char jelly[MAX_BOARD][MAX_BOARD];
    unsigned char frosting[MAX_BOARD][MAX_BOARD];
} CandyCrush;

static const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 255};
static const Color GRID_COLOR = (Color){24, 64, 64, 255};
static const Color JELLY_COLOR = (Color){255, 170, 200, 180};
static const Color FROSTING_COLOR = (Color){210, 230, 255, 255};
static const Color INGREDIENT_COLOR = (Color){201, 142, 45, 255};
static const Color CANDY_COLORS[MAX_CANDIES + 1] = {
    {20, 20, 20, 255}, {231, 76, 60, 255}, {46, 204, 113, 255}, {52, 152, 219, 255},
    {241, 196, 15, 255}, {155, 89, 182, 255}, {26, 188, 156, 255}, {230, 126, 34, 255},
    {236, 240, 241, 255},
};
static const char CANDY_SYMBOLS[MAX_CANDIES + 1] = {'.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};

static inline int max_int(int a, int b) { return a > b ? a : b; }
static inline int min_int(int a, int b) { return a < b ? a : b; }
static inline int clamp_int(int value, int low, int high) { return min_int(high, max_int(low, value)); }
static inline bool in_bounds(CandyCrush* env, int row, int col) { return row >= 0 && row < env->board_size && col >= 0 && col < env->board_size; }
static inline unsigned char make_cell(int color, SpecialType special) { return (unsigned char)(((int)special << TYPE_SHIFT) | (color & COLOR_MASK)); }
static inline int cell_color(unsigned char cell) { return cell & COLOR_MASK; }
static inline SpecialType cell_special(unsigned char cell) { return (SpecialType)((cell >> TYPE_SHIFT) & COLOR_MASK); }
static inline bool is_empty(unsigned char cell) { return cell == 0; }
static inline bool is_ingredient(unsigned char cell) { return cell_special(cell) == SPECIAL_INGREDIENT; }
static inline int match_color(unsigned char cell) {
    const SpecialType special = cell_special(cell);
    return (!is_empty(cell) && special != SPECIAL_COLOR_BOMB && special != SPECIAL_INGREDIENT) ? cell_color(cell) : 0;
}
static inline unsigned char sample_candy(CandyCrush* env) { return make_cell(1 + rand() % env->num_candies, SPECIAL_NONE); }
static inline int obs_layers(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 8; }
static inline int action_count(CandyCrush* env) { return env->board_size * env->board_size * 4; }
static inline int obs_feature_size(CandyCrush* env) { return env->board_size * env->board_size * obs_layers(env); }
static inline int color_bomb_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies; }
static inline int jelly_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 1; }
static inline int frosting_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 2; }
static inline int ingredient_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 3; }
static inline int goal_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 4; }
static inline int steps_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 5; }
static inline int objective_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 6; }
static inline int target_color_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 7; }
static inline bool is_legal_swap(CandyCrush* env, int row, int col, int nrow, int ncol);
static inline int swap_match_color(CandyCrush* env, int row, int col, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell);
static inline float lerp_float(float start, float end, float t) { return start + (end - start) * t; }
static inline int lerp_int(int start, int end, float t) { return (int)(lerp_float((float)start, (float)end, t) + 0.5f); }

static inline float clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static inline float goal_remaining_ratio(CandyCrush* env) {
    if (env->objective_mode == GOAL_SCORE) {
        return env->score_target > 0 ? clamp01((float)max_int(0, env->score_target - env->score) / env->score_target) : 0.0f;
    }
    if (env->objective_mode == GOAL_JELLY) {
        return env->jelly_total > 0 ? clamp01((float)env->jelly_remaining / env->jelly_total) : 0.0f;
    }
    if (env->objective_mode == GOAL_INGREDIENT) {
        return env->ingredient_total > 0 ? clamp01((float)env->ingredient_remaining / env->ingredient_total) : 0.0f;
    }
    if (env->objective_mode == GOAL_COLOR) {
        return env->color_target > 0 ? clamp01((float)max_int(0, env->color_target - env->color_collected) / env->color_target) : 0.0f;
    }
    return env->frosting_target > 0 ? clamp01((float)max_int(0, env->frosting_target - env->frosting_cleared) / env->frosting_target) : 0.0f;
}

static inline bool goal_complete(CandyCrush* env) {
    if (env->objective_mode == GOAL_SCORE) return env->score >= env->score_target;
    if (env->objective_mode == GOAL_JELLY) return env->jelly_remaining <= 0;
    if (env->objective_mode == GOAL_INGREDIENT) return env->ingredient_remaining <= 0;
    if (env->objective_mode == GOAL_COLOR) return env->color_collected >= env->color_target;
    return env->frosting_cleared >= env->frosting_target;
}

static inline float curriculum_win_rate(CandyCrush* env) {
    return env->frontier_episodes > 0 ? (float)env->frontier_wins / env->frontier_episodes : 0.0f;
}

static inline float level_progress(CandyCrush* env, int level) {
    const int start = clamp_int(env->curriculum_start_level, 0, MAX_LEVEL_BANK - 1);
    const int finish = clamp_int(env->curriculum_max_level, start, MAX_LEVEL_BANK - 1);
    const int span = max_int(1, finish - start);
    return clamp01((float)(level - start) / span);
}

static void restore_base_profile(CandyCrush* env) {
    env->max_steps = env->base_max_steps;
    env->objective_mode = env->base_objective_mode;
    env->score_target = env->base_score_target;
    env->frosting_layers = env->base_frosting_layers;
    env->ingredient_target = env->base_ingredient_target;
    env->ingredient_spawn_rows = env->base_ingredient_spawn_rows;
    env->target_color = env->base_target_color;
    env->color_target = env->base_color_target;
    env->frosting_target = env->base_frosting_target;
    env->jelly_density = env->base_jelly_density;
    env->frosting_density = env->base_frosting_density;
    env->starter_striped = 0;
    env->starter_wrapped = 0;
    env->starter_color_bomb = 0;
    env->starter_fish = 0;
}

static void apply_level_profile(CandyCrush* env, int level) {
    restore_base_profile(env);
    switch (level) {
        case 0:
            env->objective_mode = GOAL_COLOR;
            env->target_color = 3;
            env->color_target = 20;
            env->max_steps = 24;
            env->starter_striped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 1:
            env->objective_mode = GOAL_COLOR;
            env->target_color = 3;
            env->color_target = 30;
            env->max_steps = 28;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 2:
            env->objective_mode = GOAL_COLOR;
            env->target_color = 3;
            env->color_target = 35;
            env->max_steps = 30;
            env->starter_striped = 2;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 3:
            env->objective_mode = GOAL_FROSTING;
            env->frosting_target = 12;
            env->max_steps = 24;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 4:
            env->objective_mode = GOAL_FROSTING;
            env->frosting_target = 20;
            env->max_steps = 28;
            env->frosting_layers = 1;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 5:
            env->objective_mode = GOAL_FROSTING;
            env->frosting_target = 30;
            env->max_steps = 32;
            env->frosting_layers = 2;
            env->starter_striped = 2;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 6:
            env->objective_mode = GOAL_JELLY;
            env->max_steps = 28;
            env->jelly_density = 0.18f;
            env->frosting_density = 0.05f;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            break;
        case 7:
            env->objective_mode = GOAL_JELLY;
            env->max_steps = 30;
            env->jelly_density = 0.28f;
            env->frosting_density = 0.08f;
            env->frosting_layers = 2;
            env->starter_wrapped = 1;
            break;
        case 8:
            env->objective_mode = GOAL_JELLY;
            env->max_steps = 32;
            env->jelly_density = 0.36f;
            env->frosting_density = 0.10f;
            env->frosting_layers = 2;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            break;
        case 9:
            env->objective_mode = GOAL_INGREDIENT;
            env->ingredient_target = 1;
            env->ingredient_spawn_rows = 1;
            env->max_steps = 28;
            env->frosting_density = 0.06f;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            break;
        case 10:
            env->objective_mode = GOAL_INGREDIENT;
            env->ingredient_target = 2;
            env->ingredient_spawn_rows = 2;
            env->max_steps = 32;
            env->frosting_density = 0.10f;
            env->frosting_layers = 2;
            env->starter_wrapped = 1;
            break;
        case 11:
        default:
            env->objective_mode = GOAL_JELLY;
            env->max_steps = 34;
            env->jelly_density = 0.42f;
            env->frosting_density = 0.12f;
            env->frosting_layers = 2;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            env->starter_color_bomb = 1;
            break;
    }
}

static int select_active_level(CandyCrush* env) {
    if (env->level_id >= 0) return clamp_int(env->level_id, env->curriculum_start_level, env->curriculum_max_level);
    if (env->curriculum_mode == 0) return -1;
    if (env->unlocked_level > env->curriculum_start_level
        && ((float)rand() / RAND_MAX) < env->curriculum_replay_prob) {
        const int replay_count = env->unlocked_level - env->curriculum_start_level;
        return env->curriculum_start_level + rand() % replay_count;
    }
    return env->frontier_level;
}

static void maybe_advance_curriculum(CandyCrush* env) {
    if (env->curriculum_mode == 0 || env->level_id >= 0) return;
    if (env->frontier_episodes < env->curriculum_min_episodes) return;
    if (curriculum_win_rate(env) < env->curriculum_threshold) return;
    if (env->unlocked_level >= env->curriculum_max_level) return;
    env->unlocked_level++;
    env->frontier_level = env->unlocked_level;
    env->frontier_episodes = 0;
    env->frontier_wins = 0;
}

static void record_curriculum_result(CandyCrush* env, bool won) {
    if (env->curriculum_mode == 0 || env->level_id >= 0) return;
    if (env->active_level != env->frontier_level) return;
    env->frontier_episodes++;
    if (won) env->frontier_wins++;
    maybe_advance_curriculum(env);
}

static inline int obs_layer(CandyCrush* env, unsigned char cell) {
    const int color = cell_color(cell);
    const SpecialType special = cell_special(cell);
    if (special == SPECIAL_COLOR_BOMB) return color_bomb_layer(env);
    if (special == SPECIAL_INGREDIENT) return ingredient_layer(env);
    if (color <= 0 || color > env->num_candies) return -1;
    if (special == SPECIAL_NONE) return color - 1;
    if (special == SPECIAL_STRIPED_H) return env->num_candies + color - 1;
    if (special == SPECIAL_STRIPED_V) return 2 * env->num_candies + color - 1;
    if (special == SPECIAL_WRAPPED) return 3 * env->num_candies + color - 1;
    return 4 * env->num_candies + color - 1;
}

static void update_observations(CandyCrush* env) {
    const int cells = env->board_size * env->board_size;
    const int layers = obs_layers(env);
    const int feature_size = cells * layers;
    memset(env->observations, 0, feature_size + action_count(env));
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        const int idx = row * env->board_size + col;
        const int layer = obs_layer(env, env->board[row][col]);
        if (layer >= 0) env->observations[layer * cells + idx] = 1;
        if (env->jelly[row][col] > 0) env->observations[jelly_layer(env) * cells + idx] = 255;
        if (env->frosting[row][col] > 0) env->observations[frosting_layer(env) * cells + idx] = (unsigned char)(255 * env->frosting[row][col] / max_int(1, env->frosting_layers));
    }
    {
        const unsigned char goal = (unsigned char)(255 * goal_remaining_ratio(env));
        const unsigned char steps = (unsigned char)(255 * max_int(0, env->max_steps - env->steps) / max_int(1, env->max_steps));
        const unsigned char objective = (unsigned char)(255 * env->objective_mode / GOAL_FROSTING);
        const unsigned char target_color = env->objective_mode == GOAL_COLOR
            ? (unsigned char)(255 * env->target_color / max_int(1, env->num_candies))
            : 0;
        for (int idx = 0; idx < cells; idx++) {
            env->observations[goal_layer(env) * cells + idx] = goal;
            env->observations[steps_layer(env) * cells + idx] = steps;
            env->observations[objective_layer(env) * cells + idx] = objective;
            env->observations[target_color_layer(env) * cells + idx] = target_color;
        }
    }
    {
        unsigned char* action_mask = env->observations + feature_size;
        for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
            if (col + 1 < env->board_size) {
                const unsigned char legal = is_legal_swap(env, row, col, row, col + 1) ? 255 : 0;
                action_mask[(row * env->board_size + col) * 4 + 1] = legal;
                action_mask[(row * env->board_size + (col + 1)) * 4 + 3] = legal;
            }
            if (row + 1 < env->board_size) {
                const unsigned char legal = is_legal_swap(env, row, col, row + 1, col) ? 255 : 0;
                action_mask[(row * env->board_size + col) * 4 + 2] = legal;
                action_mask[((row + 1) * env->board_size + col) * 4 + 0] = legal;
            }
        }
    }
}

static inline void push_effect(EffectQueue* q, int kind, int row, int col, int color, int count) {
    if (q->count >= MAX_EFFECTS) return;
    q->items[q->count++] = (Effect){kind, row, col, color, count};
}

static inline Effect pop_effect(EffectQueue* q) { return q->items[--q->count]; }

static int random_existing_color(CandyCrush* env) {
    int colors[MAX_CANDIES], count = 0;
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        const int color = match_color(env->board[row][col]);
        if (color <= 0) continue;
        bool seen = false;
        for (int i = 0; i < count; i++) if (colors[i] == color) seen = true;
        if (!seen) colors[count++] = color;
    }
    return count == 0 ? 1 + rand() % env->num_candies : colors[rand() % count];
}

static bool pick_random_target(CandyCrush* env, bool clear[MAX_BOARD][MAX_BOARD], bool keep[MAX_BOARD][MAX_BOARD], int* out_row, int* out_col) {
    int rows[MAX_COMPONENTS], cols[MAX_COMPONENTS], count = 0;
    for (int pass = 0; pass < 4; pass++) {
        count = 0;
        for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
            if (keep[row][col]) continue;
            if (pass == 0 && env->jelly[row][col] > 0 && env->frosting[row][col] == 0 && !is_empty(env->board[row][col]) && !clear[row][col]) { rows[count] = row; cols[count] = col; count++; }
            else if (pass == 1 && env->frosting[row][col] > 0 && !clear[row][col]) { rows[count] = row; cols[count] = col; count++; }
            else if (pass == 2 && !is_empty(env->board[row][col]) && !is_ingredient(env->board[row][col]) && !clear[row][col]) { rows[count] = row; cols[count] = col; count++; }
            else if (pass == 3 && !is_empty(env->board[row][col]) && !is_ingredient(env->board[row][col])) { rows[count] = row; cols[count] = col; count++; }
        }
        if (count > 0) {
            const int idx = rand() % count;
            *out_row = rows[idx];
            *out_col = cols[idx];
            return true;
        }
    }
    return false;
}

static void activate_cell(
    CandyCrush* env, EffectQueue* cur, EffectQueue* post,
    bool clear[MAX_BOARD][MAX_BOARD], bool keep[MAX_BOARD][MAX_BOARD],
    bool activated[MAX_BOARD][MAX_BOARD], int row, int col
) {
    if (!in_bounds(env, row, col) || keep[row][col]) return;
    if (env->frosting[row][col] > 0) { clear[row][col] = true; return; }
    if (is_empty(env->board[row][col])) return;
    if (is_ingredient(env->board[row][col])) return;
    clear[row][col] = true;
    if (activated[row][col]) return;
    activated[row][col] = true;
    switch (cell_special(env->board[row][col])) {
        case SPECIAL_STRIPED_H: push_effect(cur, EFFECT_ROW, row, col, 0, 0); break;
        case SPECIAL_STRIPED_V: push_effect(cur, EFFECT_COL, row, col, 0, 0); break;
        case SPECIAL_WRAPPED:
            push_effect(cur, EFFECT_BLAST3, row, col, 0, 0);
            push_effect(post, EFFECT_BLAST3, row, col, 0, 0);
            break;
        case SPECIAL_COLOR_BOMB: push_effect(cur, EFFECT_COLOR, row, col, random_existing_color(env), 0); break;
        case SPECIAL_FISH: push_effect(post, EFFECT_FISH, row, col, 0, 3); break;
        case SPECIAL_NONE:
        default: break;
    }
}

static void process_effects(
    CandyCrush* env, EffectQueue* cur, EffectQueue* post,
    bool clear[MAX_BOARD][MAX_BOARD], bool keep[MAX_BOARD][MAX_BOARD],
    bool activated[MAX_BOARD][MAX_BOARD]
) {
    while (cur->count > 0) {
        const Effect e = pop_effect(cur);
        if (e.kind == EFFECT_ROW || e.kind == EFFECT_COL) {
            for (int i = 0; i < env->board_size; i++) activate_cell(env, cur, post, clear, keep, activated, e.kind == EFFECT_ROW ? e.row : i, e.kind == EFFECT_ROW ? i : e.col);
            continue;
        }
        if (e.kind == EFFECT_BLAST3 || e.kind == EFFECT_BLAST5) {
            const int radius = e.kind == EFFECT_BLAST3 ? 1 : 2;
            for (int row = e.row - radius; row <= e.row + radius; row++) for (int col = e.col - radius; col <= e.col + radius; col++) activate_cell(env, cur, post, clear, keep, activated, row, col);
            continue;
        }
        if (e.kind == EFFECT_CROSS3) {
            for (int d = -1; d <= 1; d++) {
                const int row = e.row + d, col = e.col + d;
                if (in_bounds(env, row, e.col)) for (int c = 0; c < env->board_size; c++) activate_cell(env, cur, post, clear, keep, activated, row, c);
                if (in_bounds(env, e.row, col)) for (int r = 0; r < env->board_size; r++) activate_cell(env, cur, post, clear, keep, activated, r, col);
            }
            continue;
        }
        if (e.kind == EFFECT_COLOR || e.kind == EFFECT_BOARD) {
            for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) if (e.kind == EFFECT_BOARD || match_color(env->board[row][col]) == e.color) activate_cell(env, cur, post, clear, keep, activated, row, col);
            continue;
        }
        for (int i = 0; i < max_int(1, e.count); i++) {
            int row = 0, col = 0;
            if (!pick_random_target(env, clear, keep, &row, &col)) break;
            activate_cell(env, cur, post, clear, keep, activated, row, col);
            if (e.kind == EFFECT_FISH_H) push_effect(cur, EFFECT_ROW, row, col, 0, 0);
            else if (e.kind == EFFECT_FISH_V) push_effect(cur, EFFECT_COL, row, col, 0, 0);
            else if (e.kind == EFFECT_FISH_W) push_effect(cur, EFFECT_BLAST5, row, col, 0, 0);
        }
    }
}

static void analyze_matches(CandyCrush* env, MatchAnalysis* a) {
    memset(a, 0, sizeof(MatchAnalysis));
    for (int row = 0; row < MAX_BOARD; row++) for (int col = 0; col < MAX_BOARD; col++) a->component[row][col] = -1;

    for (int row = 0; row < env->board_size; row++) for (int start = 0; start < env->board_size;) {
        const int color = match_color(env->board[row][start]);
        int end = start + 1;
        while (end < env->board_size && color > 0 && match_color(env->board[row][end]) == color) end++;
        if (color > 0 && end - start >= 3) for (int col = start; col < end; col++) a->matched[row][col] = true, a->h[row][col] = end - start;
        start = end;
    }
    for (int col = 0; col < env->board_size; col++) for (int start = 0; start < env->board_size;) {
        const int color = match_color(env->board[start][col]);
        int end = start + 1;
        while (end < env->board_size && color > 0 && match_color(env->board[end][col]) == color) end++;
        if (color > 0 && end - start >= 3) for (int row = start; row < end; row++) a->matched[row][col] = true, a->v[row][col] = end - start;
        start = end;
    }
    for (int row = 0; row < env->board_size - 1; row++) for (int col = 0; col < env->board_size - 1; col++) {
        const int color = match_color(env->board[row][col]);
        if (color > 0 && match_color(env->board[row][col + 1]) == color && match_color(env->board[row + 1][col]) == color && match_color(env->board[row + 1][col + 1]) == color) {
            a->square[row][col] = a->square[row][col + 1] = a->square[row + 1][col] = a->square[row + 1][col + 1] = true;
            a->matched[row][col] = a->matched[row][col + 1] = a->matched[row + 1][col] = a->matched[row + 1][col + 1] = true;
        }
    }
    {
        int qr[MAX_COMPONENTS], qc[MAX_COMPONENTS], drow[4] = {-1, 1, 0, 0}, dcol[4] = {0, 0, -1, 1};
        for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
            if (!a->matched[row][col] || a->component[row][col] != -1) continue;
            int head = 0, tail = 0, color = match_color(env->board[row][col]);
            qr[tail] = row; qc[tail++] = col; a->component[row][col] = a->components; a->component_color[a->components] = color;
            while (head < tail) {
                const int cr = qr[head], cc = qc[head++];
                for (int i = 0; i < 4; i++) {
                    const int nr = cr + drow[i], nc = cc + dcol[i];
                    if (!in_bounds(env, nr, nc) || !a->matched[nr][nc] || a->component[nr][nc] != -1 || match_color(env->board[nr][nc]) != color) continue;
                    a->component[nr][nc] = a->components;
                    qr[tail] = nr; qc[tail++] = nc;
                }
            }
            a->components++;
        }
    }
}

static bool has_matches(CandyCrush* env, MatchAnalysis* a) {
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) if (a->matched[row][col]) return true;
    return false;
}

static void pick_creation(
    CandyCrush* env, MatchAnalysis* a, int component, bool prefer, int pref_row, int pref_col, int move_dir,
    bool keep[MAX_BOARD][MAX_BOARD], unsigned char create[MAX_BOARD][MAX_BOARD]
) {
    int best_row = -1, best_col = -1, color = a->component_color[component], priority = -1;
    for (int pass = 0; pass < 2; pass++) for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (a->component[row][col] != component) continue;
        if (pass == 0 && (!prefer || row != pref_row || col != pref_col)) continue;
        int here = -1;
        if (a->h[row][col] >= 5 || a->v[row][col] >= 5) here = 3;
        else if (a->h[row][col] >= 3 && a->v[row][col] >= 3) here = 2;
        else if (a->square[row][col]) here = 1;
        else if (a->h[row][col] >= 4 || a->v[row][col] >= 4) here = 0;
        if (here > priority) { priority = here; best_row = row; best_col = col; }
    }
    if (priority < 0) return;
    keep[best_row][best_col] = true;
    if (priority == 3) create[best_row][best_col] = make_cell(0, SPECIAL_COLOR_BOMB);
    else if (priority == 2) create[best_row][best_col] = make_cell(color, SPECIAL_WRAPPED);
    else if (priority == 1) create[best_row][best_col] = make_cell(color, SPECIAL_FISH);
    else {
        SpecialType stripe = SPECIAL_STRIPED_H;
        if (prefer && best_row == pref_row && best_col == pref_col) stripe = (move_dir == 0 || move_dir == 2) ? SPECIAL_STRIPED_V : SPECIAL_STRIPED_H;
        else if (a->v[best_row][best_col] >= 4 && a->h[best_row][best_col] < 4) stripe = SPECIAL_STRIPED_V;
        create[best_row][best_col] = make_cell(color, stripe);
    }
}

static inline bool any_clear(CandyCrush* env, bool clear[MAX_BOARD][MAX_BOARD]) {
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) if (clear[row][col]) return true;
    return false;
}

static ClearStats apply_clear(CandyCrush* env, bool clear[MAX_BOARD][MAX_BOARD], unsigned char create[MAX_BOARD][MAX_BOARD]) {
    ClearStats stats = {0};
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (!clear[row][col]) continue;
        if (env->frosting[row][col] > 0) { env->frosting[row][col]--; stats.frosting++; continue; }
        if (is_empty(env->board[row][col])) continue;
        if (env->objective_mode == GOAL_COLOR && match_color(env->board[row][col]) == env->target_color) stats.color++;
        env->board[row][col] = 0;
        stats.candies++;
        if (env->jelly[row][col] > 0) { env->jelly[row][col] = 0; env->jelly_remaining--; stats.jelly++; }
    }
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) if (create[row][col] != 0) env->board[row][col] = create[row][col];
    return stats;
}

static int drop_ingredients(CandyCrush* env) {
    int dropped = 0;
    const int bottom = env->board_size - 1;
    for (int col = 0; col < env->board_size; col++) {
        if (env->frosting[bottom][col] > 0 || !is_ingredient(env->board[bottom][col])) continue;
        env->board[bottom][col] = 0;
        env->ingredient_remaining--;
        env->ingredients_dropped++;
        dropped++;
    }
    return dropped;
}

static void apply_gravity(CandyCrush* env) {
    for (int col = 0; col < env->board_size; col++) {
        int write_row = env->board_size - 1;
        while (write_row >= 0 && env->frosting[write_row][col] > 0) { env->board[write_row][col] = 0; write_row--; }
        for (int row = env->board_size - 1; row >= 0; row--) {
            if (env->frosting[row][col] > 0) {
                env->board[row][col] = 0;
                write_row = row - 1;
                while (write_row >= 0 && env->frosting[write_row][col] > 0) { env->board[write_row][col] = 0; write_row--; }
                continue;
            }
            if (!is_empty(env->board[row][col])) {
                if (row != write_row) { env->board[write_row][col] = env->board[row][col]; env->board[row][col] = 0; }
                write_row--;
                while (write_row >= 0 && env->frosting[write_row][col] > 0) { env->board[write_row][col] = 0; write_row--; }
            }
        }
        while (write_row >= 0) { if (env->frosting[write_row][col] == 0) env->board[write_row][col] = 0; write_row--; }
    }
}

static void refill_board(CandyCrush* env) {
    for (int col = 0; col < env->board_size; col++) for (int row = 0; row < env->board_size; row++) if (env->frosting[row][col] == 0 && is_empty(env->board[row][col])) env->board[row][col] = sample_candy(env);
}

static float resolve_board(CandyCrush* env, EffectQueue* seed, bool prefer, int pref_row, int pref_col, int move_dir) {
    EffectQueue cur = {0}, post = {0};
    float reward = 0.0f;
    int combo = 0;
    bool used_prefer = false;
    if (seed != NULL) cur = *seed;
    while (true) {
        bool clear[MAX_BOARD][MAX_BOARD] = {{0}}, keep[MAX_BOARD][MAX_BOARD] = {{0}}, activated[MAX_BOARD][MAX_BOARD] = {{0}};
        unsigned char create[MAX_BOARD][MAX_BOARD] = {{0}};
        if (cur.count == 0) {
            MatchAnalysis a;
            analyze_matches(env, &a);
            if (!has_matches(env, &a)) break;
            for (int c = 0; c < a.components; c++) pick_creation(env, &a, c, prefer && !used_prefer, pref_row, pref_col, move_dir, keep, create);
            for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) if (a.matched[row][col] && !keep[row][col]) activate_cell(env, &cur, &post, clear, keep, activated, row, col);
            process_effects(env, &cur, &post, clear, keep, activated);
            used_prefer = true;
        } else {
            process_effects(env, &cur, &post, clear, keep, activated);
        }
        if (!any_clear(env, clear)) {
            if (post.count == 0) break;
            cur = post; post.count = 0; continue;
        }
        combo++;
        {
            ClearStats stats = apply_clear(env, clear, create);
            int dropped = 0;
            env->score += stats.candies;
            env->total_cleared += stats.candies;
            env->jelly_cleared += stats.jelly;
            env->frosting_cleared += stats.frosting;
            env->color_collected += stats.color;
            apply_gravity(env);
            while ((dropped = drop_ingredients(env)) > 0) {
                stats.ingredients += dropped;
                apply_gravity(env);
            }
            reward += stats.candies * env->reward_per_tile
                + stats.jelly * env->jelly_reward
                + stats.frosting * env->frosting_reward
                + stats.ingredients * env->ingredient_reward;
            if (combo > 1) reward += (combo - 1) * env->combo_bonus;
            refill_board(env);
        }
        if (post.count > 0) { cur = post; post.count = 0; } else cur.count = 0;
    }
    env->max_combo = max_int(env->max_combo, combo);
    env->total_cascades += max_int(0, combo - 1);
    return reward;
}

static inline bool auto_swap(unsigned char a, unsigned char b) {
    if (is_empty(a) || is_empty(b)) return false;
    {
        const SpecialType sa = cell_special(a), sb = cell_special(b);
        return sa == SPECIAL_COLOR_BOMB || sb == SPECIAL_COLOR_BOMB || sa == SPECIAL_FISH || sb == SPECIAL_FISH || (sa != SPECIAL_NONE && sb != SPECIAL_NONE);
    }
}

static inline void swap_cells(CandyCrush* env, int row, int col, int nrow, int ncol) {
    const unsigned char tmp = env->board[row][col];
    env->board[row][col] = env->board[nrow][ncol];
    env->board[nrow][ncol] = tmp;
}

static float resolve_special_swap(CandyCrush* env, int row, int col, int nrow, int ncol, unsigned char first, unsigned char second) {
    EffectQueue q = {0};
    const SpecialType sa = cell_special(first), sb = cell_special(second);
    const int other_row = row, other_col = col, moved_row = nrow, moved_col = ncol;
    if (sa == SPECIAL_COLOR_BOMB && sb == SPECIAL_COLOR_BOMB) {
        env->board[other_row][other_col] = 0; env->board[moved_row][moved_col] = 0;
        push_effect(&q, EFFECT_BOARD, moved_row, moved_col, 0, 0);
        return resolve_board(env, &q, false, 0, 0, 0);
    }
    if (sa == SPECIAL_COLOR_BOMB || sb == SPECIAL_COLOR_BOMB) {
        const unsigned char other = sa == SPECIAL_COLOR_BOMB ? second : first;
        const SpecialType so = cell_special(other);
        const int color = match_color(other);
        env->board[sa == SPECIAL_COLOR_BOMB ? moved_row : other_row][sa == SPECIAL_COLOR_BOMB ? moved_col : other_col] = 0;
        if (so == SPECIAL_STRIPED_H || so == SPECIAL_STRIPED_V) {
            for (int r = 0; r < env->board_size; r++) for (int c = 0; c < env->board_size; c++) if (match_color(env->board[r][c]) == color) {
                const SpecialType stripe = ((r + c) % 2 == 0) ? SPECIAL_STRIPED_H : SPECIAL_STRIPED_V;
                env->board[r][c] = make_cell(color, stripe);
                push_effect(&q, stripe == SPECIAL_STRIPED_H ? EFFECT_ROW : EFFECT_COL, r, c, 0, 0);
            }
        } else if (so == SPECIAL_WRAPPED) {
            for (int r = 0; r < env->board_size; r++) for (int c = 0; c < env->board_size; c++) if (match_color(env->board[r][c]) == color) {
                env->board[r][c] = make_cell(color, SPECIAL_WRAPPED);
                push_effect(&q, EFFECT_BLAST3, r, c, 0, 0);
                push_effect(&q, EFFECT_BLAST3, r, c, 0, 0);
            }
        } else if (so == SPECIAL_FISH) {
            for (int r = 0; r < env->board_size; r++) for (int c = 0; c < env->board_size; c++) if (match_color(env->board[r][c]) == color) {
                env->board[r][c] = make_cell(color, SPECIAL_FISH);
                push_effect(&q, EFFECT_FISH, r, c, 0, 1);
            }
        } else push_effect(&q, EFFECT_COLOR, moved_row, moved_col, color, 0);
        return resolve_board(env, &q, false, 0, 0, 0);
    }
    env->board[other_row][other_col] = 0; env->board[moved_row][moved_col] = 0;
    if ((sa == SPECIAL_WRAPPED && sb == SPECIAL_WRAPPED) || (sa == SPECIAL_WRAPPED && sb == SPECIAL_FISH) || (sa == SPECIAL_FISH && sb == SPECIAL_WRAPPED)) {
        push_effect(&q, (sa == SPECIAL_FISH || sb == SPECIAL_FISH) ? EFFECT_FISH_W : EFFECT_BLAST5, moved_row, moved_col, 0, (sa == SPECIAL_FISH || sb == SPECIAL_FISH) ? 1 : 0);
        if (sa != SPECIAL_FISH && sb != SPECIAL_FISH) push_effect(&q, EFFECT_BLAST5, moved_row, moved_col, 0, 0);
    } else if ((sa == SPECIAL_STRIPED_H || sa == SPECIAL_STRIPED_V) && (sb == SPECIAL_STRIPED_H || sb == SPECIAL_STRIPED_V)) {
        push_effect(&q, EFFECT_ROW, moved_row, moved_col, 0, 0);
        push_effect(&q, EFFECT_COL, moved_row, moved_col, 0, 0);
    } else if (((sa == SPECIAL_STRIPED_H || sa == SPECIAL_STRIPED_V) && sb == SPECIAL_WRAPPED) || ((sb == SPECIAL_STRIPED_H || sb == SPECIAL_STRIPED_V) && sa == SPECIAL_WRAPPED)) {
        push_effect(&q, EFFECT_CROSS3, moved_row, moved_col, 0, 0);
    } else if (((sa == SPECIAL_STRIPED_H || sa == SPECIAL_STRIPED_V) && sb == SPECIAL_FISH) || ((sb == SPECIAL_STRIPED_H || sb == SPECIAL_STRIPED_V) && sa == SPECIAL_FISH)) {
        const SpecialType stripe = sa == SPECIAL_FISH ? sb : sa;
        push_effect(&q, stripe == SPECIAL_STRIPED_H ? EFFECT_FISH_H : EFFECT_FISH_V, moved_row, moved_col, 0, 1);
    } else push_effect(&q, EFFECT_FISH, moved_row, moved_col, 0, 3);
    return resolve_board(env, &q, false, 0, 0, 0);
}

static inline bool line_match_at(CandyCrush* env, int row, int col) {
    const int color = match_color(env->board[row][col]);
    int count = 1;
    if (color <= 0) return false;
    for (int c = col - 1; c >= 0 && match_color(env->board[row][c]) == color; c--) count++;
    for (int c = col + 1; c < env->board_size && match_color(env->board[row][c]) == color; c++) count++;
    if (count >= 3) return true;
    count = 1;
    for (int r = row - 1; r >= 0 && match_color(env->board[r][col]) == color; r--) count++;
    for (int r = row + 1; r < env->board_size && match_color(env->board[r][col]) == color; r++) count++;
    return count >= 3;
}

static inline bool square_match_at(CandyCrush* env, int row, int col) {
    const int color = match_color(env->board[row][col]);
    if (color <= 0) return false;
    for (int dr = -1; dr <= 0; dr++) for (int dc = -1; dc <= 0; dc++) {
        const int r0 = row + dr, c0 = col + dc;
        if (!in_bounds(env, r0, c0) || !in_bounds(env, r0 + 1, c0 + 1)) continue;
        if (match_color(env->board[r0][c0]) == color
            && match_color(env->board[r0][c0 + 1]) == color
            && match_color(env->board[r0 + 1][c0]) == color
            && match_color(env->board[r0 + 1][c0 + 1]) == color) return true;
    }
    return false;
}

static inline bool local_match_at(CandyCrush* env, int row, int col) {
    return line_match_at(env, row, col) || square_match_at(env, row, col);
}

static inline bool swap_creates_match(CandyCrush* env, int row, int col, int nrow, int ncol) {
    return local_match_at(env, row, col) || local_match_at(env, nrow, ncol);
}

static inline bool swappable_cell(CandyCrush* env, int row, int col);

static inline int swap_match_color(CandyCrush* env, int row, int col, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell) {
    if (row == srow && col == scol) return match_color(scell);
    if (row == trow && col == tcol) return match_color(tcell);
    return match_color(env->board[row][col]);
}

static inline bool swap_color_eq(CandyCrush* env, int row, int col, int color, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell) {
    return in_bounds(env, row, col)
        && swap_match_color(env, row, col, srow, scol, scell, trow, tcol, tcell) == color;
}

static inline bool line_match_after_swap(CandyCrush* env, int row, int col, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell) {
    const int color = swap_match_color(env, row, col, srow, scol, scell, trow, tcol, tcell);
    if (color <= 0) return false;
    return (swap_color_eq(env, row, col - 1, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row, col - 2, color, srow, scol, scell, trow, tcol, tcell))
        || (swap_color_eq(env, row, col - 1, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row, col + 1, color, srow, scol, scell, trow, tcol, tcell))
        || (swap_color_eq(env, row, col + 1, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row, col + 2, color, srow, scol, scell, trow, tcol, tcell))
        || (swap_color_eq(env, row - 1, col, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row - 2, col, color, srow, scol, scell, trow, tcol, tcell))
        || (swap_color_eq(env, row - 1, col, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row + 1, col, color, srow, scol, scell, trow, tcol, tcell))
        || (swap_color_eq(env, row + 1, col, color, srow, scol, scell, trow, tcol, tcell)
            && swap_color_eq(env, row + 2, col, color, srow, scol, scell, trow, tcol, tcell));
}

static inline bool square_match_after_swap(CandyCrush* env, int row, int col, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell) {
    const int color = swap_match_color(env, row, col, srow, scol, scell, trow, tcol, tcell);
    if (color <= 0) return false;
    for (int dr = -1; dr <= 0; dr++) for (int dc = -1; dc <= 0; dc++) {
        const int r0 = row + dr, c0 = col + dc;
        if (!in_bounds(env, r0, c0) || !in_bounds(env, r0 + 1, c0 + 1)) continue;
        if (swap_match_color(env, r0, c0, srow, scol, scell, trow, tcol, tcell) == color
            && swap_match_color(env, r0, c0 + 1, srow, scol, scell, trow, tcol, tcell) == color
            && swap_match_color(env, r0 + 1, c0, srow, scol, scell, trow, tcol, tcell) == color
            && swap_match_color(env, r0 + 1, c0 + 1, srow, scol, scell, trow, tcol, tcell) == color) return true;
    }
    return false;
}

static inline bool is_legal_swap(CandyCrush* env, int row, int col, int nrow, int ncol) {
    if (!swappable_cell(env, row, col) || !swappable_cell(env, nrow, ncol)) return false;
    if (auto_swap(env->board[row][col], env->board[nrow][ncol])) return true;
    return line_match_after_swap(env, row, col, row, col, env->board[nrow][ncol], nrow, ncol, env->board[row][col])
        || square_match_after_swap(env, row, col, row, col, env->board[nrow][ncol], nrow, ncol, env->board[row][col])
        || line_match_after_swap(env, nrow, ncol, row, col, env->board[nrow][ncol], nrow, ncol, env->board[row][col])
        || square_match_after_swap(env, nrow, ncol, row, col, env->board[nrow][ncol], nrow, ncol, env->board[row][col]);
}

static inline bool swappable_cell(CandyCrush* env, int row, int col) {
    return in_bounds(env, row, col) && env->frosting[row][col] == 0 && !is_empty(env->board[row][col]) && !is_ingredient(env->board[row][col]);
}

static bool has_legal_moves(CandyCrush* env) {
    const int dr[2] = {0, 1}, dc[2] = {1, 0};
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) for (int i = 0; i < 2; i++) {
        const int nr = row + dr[i], nc = col + dc[i];
        if (in_bounds(env, nr, nc) && is_legal_swap(env, row, col, nr, nc)) return true;
    }
    return false;
}

static inline bool creates_start_pattern(CandyCrush* env, int row, int col, unsigned char candy) {
    const int color = cell_color(candy);
    return (col >= 2 && match_color(env->board[row][col - 1]) == color && match_color(env->board[row][col - 2]) == color)
        || (row >= 2 && match_color(env->board[row - 1][col]) == color && match_color(env->board[row - 2][col]) == color)
        || (row >= 1 && col >= 1 && match_color(env->board[row - 1][col]) == color && match_color(env->board[row][col - 1]) == color && match_color(env->board[row - 1][col - 1]) == color);
}

static void place_ingredients(CandyCrush* env) {
    int rows[MAX_COMPONENTS], cols[MAX_COMPONENTS], count = 0;
    const int spawn_rows = min_int(env->board_size, max_int(1, env->ingredient_spawn_rows));
    env->ingredient_total = 0;
    env->ingredient_remaining = 0;
    if (env->objective_mode != GOAL_INGREDIENT || env->ingredient_target <= 0) return;
    for (int row = 0; row < spawn_rows; row++) for (int col = 0; col < env->board_size; col++) {
        if (env->frosting[row][col] > 0) continue;
        rows[count] = row;
        cols[count] = col;
        count++;
    }
    if (count == 0) {
        for (int col = 0; col < env->board_size; col++) {
            env->frosting[0][col] = 0;
            rows[count] = 0;
            cols[count] = col;
            count++;
        }
    }
    {
        const int target = min_int(env->ingredient_target, count);
        for (int i = 0; i < target; i++) {
            const int pick = i + rand() % (count - i);
            const int row = rows[pick], col = cols[pick];
            rows[pick] = rows[i];
            cols[pick] = cols[i];
            env->board[row][col] = make_cell(0, SPECIAL_INGREDIENT);
            env->ingredient_total++;
            env->ingredient_remaining++;
        }
    }
}

static void add_jelly_rect(CandyCrush* env, int row0, int col0, int height, int width) {
    for (int row = row0; row < min_int(env->board_size, row0 + height); row++) for (int col = col0; col < min_int(env->board_size, col0 + width); col++) {
        if (env->jelly[row][col] == 0) {
            env->jelly[row][col] = 1;
            env->jelly_total++;
            env->jelly_remaining++;
        }
    }
}

static void add_frosting_rect(CandyCrush* env, int row0, int col0, int height, int width, int layers) {
    for (int row = row0; row < min_int(env->board_size, row0 + height); row++) for (int col = col0; col < min_int(env->board_size, col0 + width); col++) {
        env->frosting[row][col] = max_int(env->frosting[row][col], min_int(max_int(1, layers), env->frosting_layers));
    }
}

static void build_authored_layout(CandyCrush* env) {
    switch (env->active_level) {
        case 1:
            add_frosting_rect(env, 5, 4, 2, 3, 1);
            break;
        case 2:
            add_frosting_rect(env, 4, 3, 3, 4, 1);
            break;
        case 3:
            add_frosting_rect(env, 5, 2, 2, 4, 1);
            break;
        case 4:
            add_frosting_rect(env, 4, 1, 3, 6, 1);
            break;
        case 5:
            add_frosting_rect(env, 4, 0, 4, 8, 2);
            break;
        case 6:
            add_jelly_rect(env, 2, 2, 3, 3);
            add_frosting_rect(env, 5, 5, 2, 2, 1);
            break;
        case 7:
            add_jelly_rect(env, 1, 1, 4, 4);
            add_frosting_rect(env, 5, 4, 2, 3, 2);
            break;
        case 8:
            add_jelly_rect(env, 1, 1, 6, 6);
            add_frosting_rect(env, 3, 3, 2, 2, 2);
            break;
        case 9:
            add_frosting_rect(env, 4, 2, 3, 4, 1);
            break;
        case 10:
            add_frosting_rect(env, 3, 1, 4, 6, 2);
            break;
        case 11:
            add_jelly_rect(env, 1, 1, 6, 6);
            add_frosting_rect(env, 3, 0, 2, 8, 2);
            break;
        default:
            break;
    }
}

static void randomize_layout(CandyCrush* env) {
    memset(env->jelly, 0, sizeof(env->jelly));
    memset(env->frosting, 0, sizeof(env->frosting));
    env->jelly_total = 0;
    env->jelly_remaining = 0;
    env->ingredient_total = 0;
    env->ingredient_remaining = 0;
    if (env->active_level >= 0) {
        build_authored_layout(env);
        if (env->objective_mode == GOAL_JELLY && env->jelly_total == 0) add_jelly_rect(env, 2, 2, 2, 2);
        return;
    }
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (env->objective_mode == GOAL_JELLY && ((float)rand() / RAND_MAX) < env->jelly_density) {
            env->jelly[row][col] = 1;
            env->jelly_total++;
            env->jelly_remaining++;
        }
        if (((float)rand() / RAND_MAX) < env->frosting_density) env->frosting[row][col] = 1 + rand() % max_int(1, env->frosting_layers);
    }
    if (env->objective_mode == GOAL_JELLY && env->jelly_total == 0) {
        const int row = rand() % env->board_size;
        const int col = rand() % env->board_size;
        env->jelly[row][col] = 1;
        env->jelly_total = env->jelly_remaining = 1;
    }
}

static void clear_board_preserving_blockers(CandyCrush* env) {
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) env->board[row][col] = 0;
}

static void fill_random_board(CandyCrush* env) {
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (env->frosting[row][col] > 0 || is_ingredient(env->board[row][col])) continue;
        unsigned char candy = sample_candy(env);
        for (int retry = 0; retry < 64 && creates_start_pattern(env, row, col, candy); retry++) candy = sample_candy(env);
        env->board[row][col] = candy;
    }
}

static void seed_random_specials(CandyCrush* env, SpecialType special, int count) {
    if (count <= 0) return;
    for (int placed = 0; placed < count; placed++) {
        for (int attempt = 0; attempt < 128; attempt++) {
            const int row = rand() % env->board_size;
            const int col = rand() % env->board_size;
            const unsigned char cell = env->board[row][col];
            const int color = cell_color(cell);
            if (env->frosting[row][col] > 0 || is_empty(cell) || is_ingredient(cell)) continue;
            if (special == SPECIAL_COLOR_BOMB) env->board[row][col] = make_cell(0, SPECIAL_COLOR_BOMB);
            else if (special == SPECIAL_STRIPED_H) env->board[row][col] = make_cell(max_int(1, color), rand() % 2 == 0 ? SPECIAL_STRIPED_H : SPECIAL_STRIPED_V);
            else env->board[row][col] = make_cell(max_int(1, color), special);
            break;
        }
    }
}

static void seed_starter_specials(CandyCrush* env) {
    seed_random_specials(env, SPECIAL_STRIPED_H, env->starter_striped);
    seed_random_specials(env, SPECIAL_WRAPPED, env->starter_wrapped);
    seed_random_specials(env, SPECIAL_COLOR_BOMB, env->starter_color_bomb);
    seed_random_specials(env, SPECIAL_FISH, env->starter_fish);
}

static void generate_board(CandyCrush* env) {
    for (int attempt = 0; attempt < 256; attempt++) {
        clear_board_preserving_blockers(env);
        fill_random_board(env);
        place_ingredients(env);
        seed_starter_specials(env);
        if (has_legal_moves(env)) return;
    }
}

static void reshuffle_board(CandyCrush* env) {
    for (int attempt = 0; attempt < 256; attempt++) {
        for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
            if (env->frosting[row][col] == 0 && !is_ingredient(env->board[row][col])) env->board[row][col] = 0;
        }
        fill_random_board(env);
        if (has_legal_moves(env)) return;
    }
}

static void reset_episode(CandyCrush* env) {
    env->active_level = select_active_level(env);
    if (env->active_level >= 0) apply_level_profile(env, env->active_level);
    else restore_base_profile(env);
    env->steps = env->score = env->total_cleared = env->invalid_swaps = env->successful_swaps = 0;
    env->total_cascades = env->max_combo = env->reshuffles = 0;
    env->jelly_cleared = env->frosting_cleared = env->level_won = 0;
    env->ingredients_dropped = 0;
    env->color_collected = 0;
    env->episode_return = 0.0f;
    clear_board_preserving_blockers(env);
    randomize_layout(env);
    generate_board(env);
    update_observations(env);
}

static void write_episode_log(CandyCrush* env) {
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->steps;
    env->log.total_cleared += env->total_cleared;
    env->log.invalid_swaps += env->invalid_swaps;
    env->log.successful_swaps += env->successful_swaps;
    env->log.total_cascades += env->total_cascades;
    env->log.max_combo += env->max_combo;
    env->log.reshuffles += env->reshuffles;
    env->log.jelly_cleared += env->jelly_cleared;
    env->log.frosting_cleared += env->frosting_cleared;
    env->log.ingredient_dropped += env->ingredients_dropped;
    env->log.color_collected += env->color_collected;
    env->log.goal_progress += 1.0f - goal_remaining_ratio(env);
    env->log.level_wins += env->level_won;
    env->log.level_id += env->active_level >= 0 ? env->active_level : 0.0f;
    env->log.unlocked_level += env->unlocked_level;
    env->log.curriculum_win_rate += curriculum_win_rate(env);
    env->log.n += 1.0f;
}

static inline bool decode_action(CandyCrush* env, int* row, int* col, int* nrow, int* ncol, int* dir) {
    int action = env->actions[0], count = env->board_size * env->board_size * 4;
    if (action < 0) action = 0;
    action %= count;
    *dir = action % 4;
    {
        const int cell = action / 4;
        *row = cell / env->board_size; *col = cell % env->board_size; *nrow = *row; *ncol = *col;
    }
    if (*dir == 0) (*nrow)--; else if (*dir == 1) (*ncol)++; else if (*dir == 2) (*nrow)++; else (*ncol)--;
    return in_bounds(env, *nrow, *ncol);
}

static void init_env(CandyCrush* env) {
    if (env->board_size < 4 || env->board_size > MAX_BOARD) { fprintf(stderr, "candy_crush: board_size must be in [4, %d]\n", MAX_BOARD); exit(1); }
    if (env->num_candies < 4 || env->num_candies > MAX_CANDIES) { fprintf(stderr, "candy_crush: num_candies must be in [4, %d]\n", MAX_CANDIES); exit(1); }
    if (env->max_steps < 1) { fprintf(stderr, "candy_crush: max_steps must be >= 1\n"); exit(1); }
    if (env->objective_mode < GOAL_SCORE || env->objective_mode > GOAL_FROSTING) { fprintf(stderr, "candy_crush: objective_mode must be in [0, 4]\n"); exit(1); }
    if (env->frosting_layers < 1) env->frosting_layers = 1;
    if (env->ingredient_target < 0) env->ingredient_target = 0;
    if (env->ingredient_spawn_rows < 1) env->ingredient_spawn_rows = 1;
    if (env->target_color < 0) env->target_color = 0;
    if (env->color_target < 0) env->color_target = 0;
    if (env->frosting_target < 0) env->frosting_target = 0;
    env->jelly_density = clamp01(env->jelly_density);
    env->frosting_density = clamp01(env->frosting_density);
    env->level_id = max_int(-1, env->level_id);
    env->curriculum_mode = env->curriculum_mode != 0;
    env->curriculum_start_level = clamp_int(env->curriculum_start_level, 0, MAX_LEVEL_BANK - 1);
    env->curriculum_max_level = clamp_int(env->curriculum_max_level, env->curriculum_start_level, MAX_LEVEL_BANK - 1);
    if (env->curriculum_min_episodes < 1) env->curriculum_min_episodes = 1;
    env->curriculum_threshold = clamp01(env->curriculum_threshold);
    env->curriculum_replay_prob = clamp01(env->curriculum_replay_prob);
    env->base_max_steps = env->max_steps;
    env->base_objective_mode = env->objective_mode;
    env->base_score_target = env->score_target;
    env->base_frosting_layers = env->frosting_layers;
    env->base_ingredient_target = env->ingredient_target;
    env->base_ingredient_spawn_rows = env->ingredient_spawn_rows;
    env->base_target_color = env->target_color;
    env->base_color_target = env->color_target;
    env->base_frosting_target = env->frosting_target;
    env->base_jelly_density = env->jelly_density;
    env->base_frosting_density = env->frosting_density;
    env->active_level = env->level_id >= 0 ? clamp_int(env->level_id, env->curriculum_start_level, env->curriculum_max_level) : -1;
    env->unlocked_level = env->active_level >= 0 ? env->active_level : env->curriculum_start_level;
    env->frontier_level = env->active_level >= 0 ? env->active_level : env->curriculum_start_level;
    env->frontier_episodes = 0;
    env->frontier_wins = 0;
    memset(&env->log, 0, sizeof(Log));
}

static void c_reset(CandyCrush* env) {
    if (env->terminals) env->terminals[0] = 0;
    if (env->rewards) env->rewards[0] = 0.0f;
    reset_episode(env);
}

static void c_step(CandyCrush* env) {
    int row, col, nrow, ncol, dir;
    float reward = 0.0f;
    env->rewards[0] = 0.0f; env->terminals[0] = 0; env->steps++;
    if (!decode_action(env, &row, &col, &nrow, &ncol, &dir) || !swappable_cell(env, row, col) || !swappable_cell(env, nrow, ncol)) {
        reward = env->invalid_penalty; env->invalid_swaps++;
    } else {
        const unsigned char first = env->board[row][col], second = env->board[nrow][ncol];
        swap_cells(env, row, col, nrow, ncol);
        if (auto_swap(first, second)) { env->successful_swaps++; reward = resolve_special_swap(env, row, col, nrow, ncol, first, second); }
        else if (!swap_creates_match(env, row, col, nrow, ncol)) { swap_cells(env, row, col, nrow, ncol); reward = env->invalid_penalty; env->invalid_swaps++; }
        else { env->successful_swaps++; reward = resolve_board(env, NULL, true, nrow, ncol, dir); }
        if (!goal_complete(env) && !has_legal_moves(env)) { reshuffle_board(env); env->reshuffles++; reward += env->shuffle_penalty; }
    }
    if (goal_complete(env)) {
        reward += env->success_bonus;
        env->level_won = 1;
        env->episode_return += reward;
        env->rewards[0] = reward;
        update_observations(env);
        env->terminals[0] = 1;
        record_curriculum_result(env, true);
        write_episode_log(env);
        reset_episode(env);
        env->rewards[0] = reward;
        return;
    }
    env->episode_return += reward; env->rewards[0] = reward; update_observations(env);
    if (env->steps >= env->max_steps) {
        env->terminals[0] = 1; record_curriculum_result(env, false); write_episode_log(env); reset_episode(env); env->rewards[0] = reward;
    }
}

static inline char special_marker(unsigned char cell) {
    if (cell_special(cell) == SPECIAL_STRIPED_H) return '-';
    if (cell_special(cell) == SPECIAL_STRIPED_V) return '|';
    if (cell_special(cell) == SPECIAL_WRAPPED) return 'W';
    if (cell_special(cell) == SPECIAL_COLOR_BOMB) return 'O';
    if (cell_special(cell) == SPECIAL_FISH) return 'F';
    return ' ';
}

static inline const char* goal_name(CandyCrush* env) {
    if (env->objective_mode == GOAL_SCORE) return "Score";
    if (env->objective_mode == GOAL_JELLY) return "Jelly";
    if (env->objective_mode == GOAL_INGREDIENT) return "Ingredient";
    if (env->objective_mode == GOAL_COLOR) return "Collect Color";
    return "Clear Blockers";
}

static inline const char* candy_name(CandyCrush* env, int color) {
    static const char* names[MAX_CANDIES + 1] = {
        "None", "Red", "Green", "Blue", "Yellow", "Purple", "Teal", "Orange", "White"
    };
    return names[clamp_int(color, 0, MAX_CANDIES)];
}

static void c_render(CandyCrush* env) {
    const int cell = 64, gap = 6, width = env->board_size * cell, height = env->board_size * cell + 136;
    char label[96];
    if (!IsWindowReady()) { InitWindow(width, height, "PufferLib Candy Crush"); SetTargetFPS(30); }
    if (IsKeyDown(KEY_ESCAPE)) { CloseWindow(); exit(0); }
    BeginDrawing(); ClearBackground(PUFF_BACKGROUND);
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        const unsigned char candy = env->board[row][col];
        const int color = cell_special(candy) == SPECIAL_COLOR_BOMB ? 0 : min_int(cell_color(candy), MAX_CANDIES);
        const int x = col * cell + gap / 2, y = row * cell + gap / 2;
        DrawRectangle(x, y, cell - gap, cell - gap, GRID_COLOR);
        if (env->jelly[row][col] > 0) DrawRectangleLines(x + 4, y + 4, cell - gap - 8, cell - gap - 8, JELLY_COLOR);
        if (env->frosting[row][col] > 0) {
            DrawRectangle(x + 10, y + 10, cell - gap - 20, cell - gap - 20, FROSTING_COLOR);
            snprintf(label, sizeof(label), "%d", env->frosting[row][col]); DrawText(label, x + 22, y + 18, 20, GRID_COLOR);
            continue;
        }
        if (is_ingredient(candy)) {
            DrawCircle(x + (cell - gap) / 2, y + (cell - gap) / 2, 22, INGREDIENT_COLOR);
            DrawText("I", x + 24, y + 12, 22, PUFF_WHITE);
            continue;
        }
        if (cell_special(candy) == SPECIAL_COLOR_BOMB) {
            DrawCircle(x + (cell - gap) / 2, y + (cell - gap) / 2, 22, PUFF_WHITE);
            DrawCircleLines(x + (cell - gap) / 2, y + (cell - gap) / 2, 22, CANDY_COLORS[5]);
        } else if (!is_empty(candy)) DrawCircle(x + (cell - gap) / 2, y + (cell - gap) / 2, 22, CANDY_COLORS[color]);
        snprintf(label, sizeof(label), "%c", CANDY_SYMBOLS[color]); DrawText(label, x + 22, y + 12, 22, PUFF_WHITE);
        if (special_marker(candy) != ' ') { snprintf(label, sizeof(label), "%c", special_marker(candy)); DrawText(label, x + 22, y + 32, 18, PUFF_WHITE); }
    }
    if (env->objective_mode == GOAL_COLOR) snprintf(label, sizeof(label), "Level: %d  Unlocked: %d  Goal: Collect %d %s", env->active_level >= 0 ? env->active_level : 0, env->unlocked_level, env->color_target, candy_name(env, env->target_color));
    else if (env->objective_mode == GOAL_FROSTING) snprintf(label, sizeof(label), "Level: %d  Unlocked: %d  Goal: Clear %d blockers", env->active_level >= 0 ? env->active_level : 0, env->unlocked_level, env->frosting_target);
    else snprintf(label, sizeof(label), "Level: %d  Unlocked: %d  Goal: %s", env->active_level >= 0 ? env->active_level : 0, env->unlocked_level, goal_name(env));
    DrawText(label, 12, env->board_size * cell + 8, 22, PUFF_WHITE);
    snprintf(label, sizeof(label), "Score: %d  Steps: %d/%d", env->score, env->steps, env->max_steps); DrawText(label, 12, env->board_size * cell + 34, 22, PUFF_WHITE);
    if (env->objective_mode == GOAL_COLOR) snprintf(label, sizeof(label), "%s collected: %d/%d  Frosting cleared: %d", candy_name(env, env->target_color), env->color_collected, env->color_target, env->frosting_cleared);
    else snprintf(label, sizeof(label), "Jelly: %d/%d  Frosting cleared: %d", env->jelly_cleared, env->jelly_total, env->frosting_cleared);
    DrawText(label, 12, env->board_size * cell + 60, 20, PUFF_WHITE);
    snprintf(label, sizeof(label), "Ingredients: %d/%d  Combo: %d", env->ingredients_dropped, env->ingredient_total, env->max_combo); DrawText(label, 12, env->board_size * cell + 84, 20, PUFF_WHITE);
    snprintf(label, sizeof(label), "Goal remaining: %.2f  Curriculum WR: %.2f", goal_remaining_ratio(env), curriculum_win_rate(env)); DrawText(label, 12, env->board_size * cell + 104, 20, PUFF_WHITE);
    EndDrawing();
}

static void c_close(CandyCrush* env) { if (IsWindowReady()) CloseWindow(); }

#endif
