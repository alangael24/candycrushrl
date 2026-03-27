#ifndef PUFFERLIB_CANDY_CRUSH_H
#define PUFFERLIB_CANDY_CRUSH_H

#include <stdbool.h>
#include <stdint.h>
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
#define GOAL_EXTRA_SLOTS 4
#define GOAL_AUTO -1

typedef enum { SPECIAL_NONE, SPECIAL_STRIPED_H, SPECIAL_STRIPED_V, SPECIAL_WRAPPED, SPECIAL_COLOR_BOMB, SPECIAL_FISH, SPECIAL_INGREDIENT } SpecialType;
typedef enum { EFFECT_ROW, EFFECT_COL, EFFECT_BLAST3, EFFECT_BLAST5, EFFECT_CROSS3, EFFECT_COLOR, EFFECT_BOARD, EFFECT_FISH, EFFECT_FISH_H, EFFECT_FISH_V, EFFECT_FISH_W } EffectKind;

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

typedef struct {
    int candies;
    int events[MAX_CANDIES + GOAL_EXTRA_SLOTS];
} ClearStats;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    uint64_t rng_seed;
    uint64_t rng_state;

    int board_size;
    int num_candies;
    int max_steps;
    int frosting_layers;
    int ingredient_spawn_rows;
    int task_distribution_mode;
    int task_min_active_goals;
    int task_max_active_goals;
    int task_min_steps;
    int task_max_steps;
    int goal_target[MAX_CANDIES + GOAL_EXTRA_SLOTS];
    int goal_remaining[MAX_CANDIES + GOAL_EXTRA_SLOTS];
    int has_goal_vector;

    float reward_per_tile;
    float combo_bonus;
    float invalid_penalty;
    float shuffle_penalty;
    float progress_reward_scale;
    float shaping_gamma;
    float success_bonus;
    float failure_penalty;
    float efficiency_bonus;
    float jelly_density;
    float frosting_density;

    int base_max_steps;
    int base_frosting_layers;
    int base_ingredient_spawn_rows;
    int base_goal_target[MAX_CANDIES + GOAL_EXTRA_SLOTS];
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
    int color_collected[MAX_CANDIES];
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
static inline uint64_t mix_seed64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline void seed_rng(CandyCrush* env, uint64_t seed) {
    env->rng_seed = seed;
    env->rng_state = mix_seed64(seed + 0x9E3779B97F4A7C15ULL);
    if (env->rng_state == 0) env->rng_state = 0x2545F4914F6CDD1DULL;
}
static inline void ensure_rng_seeded(CandyCrush* env) {
    if (env->rng_state == 0) seed_rng(env, env->rng_seed);
}
static inline uint32_t rng_u32(CandyCrush* env) {
    ensure_rng_seeded(env);
    uint64_t x = env->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    env->rng_state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}
static inline int rng_int_bounded(CandyCrush* env, int upper) {
    return upper <= 1 ? 0 : (int)(rng_u32(env) % (uint32_t)upper);
}
static inline float rng_unit_float(CandyCrush* env) {
    return (float)((double)rng_u32(env) / 4294967296.0);
}
static inline unsigned char sample_candy(CandyCrush* env) { return make_cell(1 + rng_int_bounded(env, env->num_candies), SPECIAL_NONE); }
static inline int goal_slot_count(CandyCrush* env) { return env->num_candies + GOAL_EXTRA_SLOTS; }
static inline int goal_jelly_slot(CandyCrush* env) { return env->num_candies; }
static inline int goal_frosting_slot(CandyCrush* env) { return env->num_candies + 1; }
static inline int goal_ingredient_slot(CandyCrush* env) { return env->num_candies + 2; }
static inline int goal_score_slot(CandyCrush* env) { return env->num_candies + 3; }
static inline int obs_layers(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 4; }
static inline int scalar_feature_count(CandyCrush* env) { return goal_slot_count(env) * 3 + 2; }
static inline int board_feature_size(CandyCrush* env) { return env->board_size * env->board_size * obs_layers(env); }
static inline int action_count(CandyCrush* env) { return env->board_size * env->board_size * 4; }
static inline int obs_feature_size(CandyCrush* env) { return board_feature_size(env) + scalar_feature_count(env); }
static inline int color_bomb_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies; }
static inline int jelly_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 1; }
static inline int frosting_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 2; }
static inline int ingredient_layer(CandyCrush* env) { return SPECIAL_LAYERS * env->num_candies + 3; }
static inline bool is_legal_swap(CandyCrush* env, int row, int col, int nrow, int ncol);
static inline int swap_match_color(CandyCrush* env, int row, int col, int srow, int scol, unsigned char scell, int trow, int tcol, unsigned char tcell);
static inline float lerp_float(float start, float end, float t) { return start + (end - start) * t; }
static inline int lerp_int(int start, int end, float t) { return (int)(lerp_float((float)start, (float)end, t) + 0.5f); }

static inline float clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static inline int rand_int_range(CandyCrush* env, int low, int high) {
    if (high < low) {
        const int tmp = low;
        low = high;
        high = tmp;
    }
    return low + rng_int_bounded(env, max_int(1, high - low + 1));
}

static inline float rand_float_range(CandyCrush* env, float low, float high) {
    if (high < low) {
        const float tmp = low;
        low = high;
        high = tmp;
    }
    return low + rng_unit_float(env) * (high - low);
}

static inline void clear_goal_vector(int goals[MAX_CANDIES + GOAL_EXTRA_SLOTS]) {
    memset(goals, 0, sizeof(int) * (MAX_CANDIES + GOAL_EXTRA_SLOTS));
}

static inline void copy_goal_vector(
    int dst[MAX_CANDIES + GOAL_EXTRA_SLOTS],
    const int src[MAX_CANDIES + GOAL_EXTRA_SLOTS]
) {
    memcpy(dst, src, sizeof(int) * (MAX_CANDIES + GOAL_EXTRA_SLOTS));
}

static inline void add_goal_event(CandyCrush* env, ClearStats* stats, int slot, int amount) {
    if (slot < 0 || slot >= goal_slot_count(env) || amount <= 0) return;
    stats->events[slot] += amount;
}

static inline void merge_clear_stats(CandyCrush* env, ClearStats* dst, const ClearStats* src) {
    dst->candies += src->candies;
    for (int i = 0; i < goal_slot_count(env); i++) dst->events[i] += src->events[i];
}

static inline void reset_goal_remaining(CandyCrush* env) {
    clear_goal_vector(env->goal_remaining);
    for (int i = 0; i < goal_slot_count(env); i++) env->goal_remaining[i] = max_int(0, env->goal_target[i]);
}

static inline void apply_goal_events(CandyCrush* env, const ClearStats* stats) {
    for (int i = 0; i < goal_slot_count(env); i++) {
        env->goal_remaining[i] = max_int(0, env->goal_remaining[i] - max_int(0, stats->events[i]));
    }
}

static inline int total_color_collected(CandyCrush* env) {
    int total = 0;
    for (int i = 0; i < env->num_candies; i++) total += env->color_collected[i];
    return total;
}

static inline bool has_score_goal(CandyCrush* env) {
    return env->goal_target[goal_score_slot(env)] > 0;
}

static inline bool has_color_goals(CandyCrush* env) {
    for (int i = 0; i < env->num_candies; i++) if (env->goal_target[i] > 0) return true;
    return false;
}

static inline bool is_score_only_goal(CandyCrush* env) {
    const int score_slot = goal_score_slot(env);
    if (env->goal_target[score_slot] <= 0) return false;
    for (int i = 0; i < goal_slot_count(env); i++) {
        if (i == score_slot) continue;
        if (env->goal_target[i] > 0) return false;
    }
    return true;
}

static inline bool has_noncolor_goals(CandyCrush* env) {
    return env->goal_target[goal_jelly_slot(env)] > 0
        || env->goal_target[goal_frosting_slot(env)] > 0
        || env->goal_target[goal_ingredient_slot(env)] > 0
        || env->goal_target[goal_score_slot(env)] > 0;
}

static inline int total_frosting_layers(CandyCrush* env) {
    int total = 0;
    for (int row = 0; row < env->board_size; row++) {
        for (int col = 0; col < env->board_size; col++) total += env->frosting[row][col];
    }
    return total;
}

static inline float goal_remaining_mass(CandyCrush* env) {
    float remaining = 0.0f;
    for (int i = 0; i < goal_slot_count(env); i++) {
        const int target = max_int(0, env->goal_target[i]);
        if (target <= 0) continue;
        remaining += clamp01((float)env->goal_remaining[i] / max_int(1, target));
    }
    return remaining;
}

static inline float goal_remaining_ratio(CandyCrush* env) {
    float remaining = 0.0f;
    int active = 0;
    for (int i = 0; i < goal_slot_count(env); i++) {
        const int target = max_int(0, env->goal_target[i]);
        if (target <= 0) continue;
        active++;
        remaining += clamp01((float)env->goal_remaining[i] / max_int(1, target));
    }
    return active > 0 ? remaining / active : 0.0f;
}

static inline float goal_potential(CandyCrush* env) {
    float phi = 0.0f;
    int active = 0;
    for (int i = 0; i < goal_slot_count(env); i++) {
        const int target = max_int(0, env->goal_target[i]);
        if (target <= 0) continue;
        phi -= clamp01((float)env->goal_remaining[i] / max_int(1, target));
        active++;
    }
    return active > 0 ? phi / active : 0.0f;
}

static inline bool goal_complete(CandyCrush* env) {
    int active = 0;
    for (int i = 0; i < goal_slot_count(env); i++) {
        if (env->goal_target[i] <= 0) continue;
        active++;
        if (env->goal_remaining[i] > 0) return false;
    }
    return active > 0;
}

static inline float curriculum_win_rate(CandyCrush* env) {
    return env->frontier_episodes > 0 ? (float)env->frontier_wins / env->frontier_episodes : 0.0f;
}

static inline float objective_tile_reward(CandyCrush* env) {
    return is_score_only_goal(env) ? env->reward_per_tile : 0.0f;
}

static inline float objective_combo_bonus(CandyCrush* env) {
    return is_score_only_goal(env) ? env->combo_bonus : 0.0f;
}

static inline float level_progress(CandyCrush* env, int level) {
    const int start = clamp_int(env->curriculum_start_level, 0, MAX_LEVEL_BANK - 1);
    const int finish = clamp_int(env->curriculum_max_level, start, MAX_LEVEL_BANK - 1);
    const int span = max_int(1, finish - start);
    return clamp01((float)(level - start) / span);
}

static void restore_base_profile(CandyCrush* env) {
    env->max_steps = env->base_max_steps;
    env->frosting_layers = env->base_frosting_layers;
    env->ingredient_spawn_rows = env->base_ingredient_spawn_rows;
    copy_goal_vector(env->goal_target, env->base_goal_target);
    env->jelly_density = env->base_jelly_density;
    env->frosting_density = env->base_frosting_density;
    env->starter_striped = 0;
    env->starter_wrapped = 0;
    env->starter_color_bomb = 0;
    env->starter_fish = 0;
}

static void apply_level_profile(CandyCrush* env, int level) {
    restore_base_profile(env);
    clear_goal_vector(env->goal_target);
    switch (level) {
        case 0:
            env->goal_target[2] = 16;
            env->max_steps = 26;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 1:
            env->goal_target[2] = 24;
            env->max_steps = 30;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 2:
            env->goal_target[2] = 30;
            env->max_steps = 32;
            env->starter_striped = 2;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 3:
            env->goal_target[goal_frosting_slot(env)] = 12;
            env->max_steps = 24;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 4:
            env->goal_target[goal_frosting_slot(env)] = 20;
            env->max_steps = 28;
            env->frosting_layers = 1;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 5:
            env->goal_target[goal_frosting_slot(env)] = 30;
            env->max_steps = 32;
            env->frosting_layers = 2;
            env->starter_striped = 2;
            env->starter_wrapped = 1;
            env->jelly_density = 0.0f;
            env->frosting_density = 0.0f;
            break;
        case 6:
            env->goal_target[goal_jelly_slot(env)] = GOAL_AUTO;
            env->max_steps = 28;
            env->jelly_density = 0.18f;
            env->frosting_density = 0.05f;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            break;
        case 7:
            env->goal_target[goal_jelly_slot(env)] = GOAL_AUTO;
            env->max_steps = 30;
            env->jelly_density = 0.28f;
            env->frosting_density = 0.08f;
            env->frosting_layers = 2;
            env->starter_wrapped = 1;
            break;
        case 8:
            env->goal_target[goal_jelly_slot(env)] = GOAL_AUTO;
            env->max_steps = 32;
            env->jelly_density = 0.36f;
            env->frosting_density = 0.10f;
            env->frosting_layers = 2;
            env->starter_striped = 1;
            env->starter_wrapped = 1;
            break;
        case 9:
            env->goal_target[goal_ingredient_slot(env)] = 1;
            env->ingredient_spawn_rows = 1;
            env->max_steps = 28;
            env->frosting_density = 0.06f;
            env->frosting_layers = 1;
            env->starter_striped = 1;
            break;
        case 10:
            env->goal_target[goal_ingredient_slot(env)] = 2;
            env->ingredient_spawn_rows = 2;
            env->max_steps = 32;
            env->frosting_density = 0.10f;
            env->frosting_layers = 2;
            env->starter_wrapped = 1;
            break;
        case 11:
        default:
            env->goal_target[goal_jelly_slot(env)] = GOAL_AUTO;
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

static void sample_task_distribution(CandyCrush* env) {
    int goal_slots[MAX_CANDIES + GOAL_EXTRA_SLOTS];
    const int jelly_slot = goal_jelly_slot(env);
    const int frosting_slot = goal_frosting_slot(env);
    const int ingredient_slot = goal_ingredient_slot(env);
    const int score_slot = goal_score_slot(env);
    const int total_goal_slots = goal_slot_count(env);
    int active_goals;

    restore_base_profile(env);
    clear_goal_vector(env->goal_target);
    env->active_level = -1;

    env->task_min_steps = max_int(1, env->task_min_steps);
    env->task_max_steps = max_int(env->task_min_steps, env->task_max_steps);
    env->max_steps = rand_int_range(env, env->task_min_steps, env->task_max_steps);

    env->jelly_density = 0.0f;
    env->frosting_density = rand_float_range(env, 0.0f, 0.12f);
    env->frosting_layers = rand_int_range(env, 1, max_int(1, env->base_frosting_layers));
    env->ingredient_spawn_rows = rand_int_range(env, 1, min_int(2, env->board_size));
    env->starter_striped = rand_int_range(env, 0, 1);
    env->starter_wrapped = rand_int_range(env, 0, 1);
    env->starter_color_bomb = rand_int_range(env, 0, 4) == 0;
    env->starter_fish = rand_int_range(env, 0, 3) == 0;

    env->task_min_active_goals = clamp_int(env->task_min_active_goals, 1, total_goal_slots);
    env->task_max_active_goals = clamp_int(env->task_max_active_goals, env->task_min_active_goals, total_goal_slots);
    active_goals = rand_int_range(env, env->task_min_active_goals, env->task_max_active_goals);

    for (int i = 0; i < total_goal_slots; i++) goal_slots[i] = i;
    for (int i = 0; i < total_goal_slots; i++) {
        const int pick = rand_int_range(env, i, total_goal_slots - 1);
        const int tmp = goal_slots[i];
        goal_slots[i] = goal_slots[pick];
        goal_slots[pick] = tmp;
    }

    for (int idx = 0; idx < active_goals; idx++) {
        const int slot = goal_slots[idx];
        if (slot < env->num_candies) {
            const int color_min = max_int(6, env->max_steps / 3);
            const int color_max = max_int(color_min, env->max_steps);
            env->goal_target[slot] = rand_int_range(env, color_min, color_max);
        } else if (slot == jelly_slot) {
            env->goal_target[jelly_slot] = GOAL_AUTO;
            env->jelly_density = rand_float_range(env, 0.12f, 0.42f);
        } else if (slot == frosting_slot) {
            env->goal_target[frosting_slot] = GOAL_AUTO;
            env->frosting_density = rand_float_range(env, 0.08f, 0.20f);
            env->frosting_layers = rand_int_range(env, 1, 2);
        } else if (slot == ingredient_slot) {
            env->goal_target[ingredient_slot] = rand_int_range(env, 1, min_int(2, env->board_size));
            env->ingredient_spawn_rows = rand_int_range(env, 1, min_int(2, env->board_size));
        } else if (slot == score_slot) {
            env->goal_target[score_slot] = rand_int_range(env, env->max_steps * 2, env->max_steps * 5);
        }
    }

    env->max_steps += 4 * max_int(0, active_goals - 1);
}

static int select_active_level(CandyCrush* env) {
    if (env->task_distribution_mode != 0 && env->level_id < 0) return -1;
    if (env->level_id >= 0) return clamp_int(env->level_id, env->curriculum_start_level, env->curriculum_max_level);
    if (env->curriculum_mode == 0) return -1;
    if (env->unlocked_level > env->curriculum_start_level
        && rng_unit_float(env) < env->curriculum_replay_prob) {
        const int replay_count = env->unlocked_level - env->curriculum_start_level;
        return env->curriculum_start_level + rng_int_bounded(env, replay_count);
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

static inline void write_board_obs(CandyCrush* env, unsigned char* board_obs) {
    const int cells = env->board_size * env->board_size;
    memset(board_obs, 0, board_feature_size(env));
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        const int idx = row * env->board_size + col;
        const int layer = obs_layer(env, env->board[row][col]);
        if (layer >= 0) board_obs[layer * cells + idx] = 255;
        if (env->jelly[row][col] > 0) board_obs[jelly_layer(env) * cells + idx] = 255;
        if (env->frosting[row][col] > 0) {
            board_obs[frosting_layer(env) * cells + idx] = (unsigned char)(
                255 * env->frosting[row][col] / max_int(1, env->frosting_layers)
            );
        }
    }
}

static inline void write_meta_obs(CandyCrush* env, unsigned char* meta_obs) {
    const int slots = goal_slot_count(env);
    const unsigned char steps = (unsigned char)(
        255 * max_int(0, env->max_steps - env->steps) / max_int(1, env->max_steps)
    );
    const unsigned char goal = (unsigned char)(255 * goal_remaining_ratio(env));
    for (int i = 0; i < slots; i++) {
        const int target = max_int(0, env->goal_target[i]);
        meta_obs[i] = (unsigned char)clamp_int(max_int(0, env->goal_target[i]), 0, 255);
        meta_obs[slots + i] = (unsigned char)clamp_int(env->goal_remaining[i], 0, 255);
        meta_obs[slots * 2 + i] = (unsigned char)(
            255.0f * clamp01((float)env->goal_remaining[i] / max_int(1, target))
        );
    }
    meta_obs[slots * 3] = steps;
    meta_obs[slots * 3 + 1] = goal;
}

static inline bool write_action_mask(CandyCrush* env, unsigned char* action_mask) {
    bool any_legal = false;
    if (action_mask != NULL) memset(action_mask, 0, action_count(env));
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (col + 1 < env->board_size) {
            const unsigned char legal = is_legal_swap(env, row, col, row, col + 1) ? 255 : 0;
            any_legal = any_legal || legal != 0;
            if (action_mask != NULL) {
                action_mask[(row * env->board_size + col) * 4 + 1] = legal;
                action_mask[(row * env->board_size + (col + 1)) * 4 + 3] = legal;
            }
        }
        if (row + 1 < env->board_size) {
            const unsigned char legal = is_legal_swap(env, row, col, row + 1, col) ? 255 : 0;
            any_legal = any_legal || legal != 0;
            if (action_mask != NULL) {
                action_mask[(row * env->board_size + col) * 4 + 2] = legal;
                action_mask[((row + 1) * env->board_size + col) * 4 + 0] = legal;
            }
        }
    }
    return any_legal;
}

static bool update_observations(CandyCrush* env) {
    unsigned char* board_obs = env->observations;
    unsigned char* meta_obs = board_obs + board_feature_size(env);
    unsigned char* action_mask = meta_obs + scalar_feature_count(env);
    write_board_obs(env, board_obs);
    write_meta_obs(env, meta_obs);
    return write_action_mask(env, action_mask);
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
    return count == 0 ? 1 + rng_int_bounded(env, env->num_candies) : colors[rng_int_bounded(env, count)];
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
            const int idx = rng_int_bounded(env, count);
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
        if (env->frosting[row][col] > 0) {
            env->frosting[row][col]--;
            add_goal_event(env, &stats, goal_frosting_slot(env), 1);
            continue;
        }
        if (is_empty(env->board[row][col])) continue;
        {
            const int color = match_color(env->board[row][col]);
            if (color > 0 && color <= env->num_candies) add_goal_event(env, &stats, color - 1, 1);
        }
        env->board[row][col] = 0;
        stats.candies++;
        if (env->jelly[row][col] > 0) {
            env->jelly[row][col] = 0;
            env->jelly_remaining--;
            add_goal_event(env, &stats, goal_jelly_slot(env), 1);
        }
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

static float resolve_board(CandyCrush* env, EffectQueue* seed, bool prefer, int pref_row, int pref_col, int move_dir, ClearStats* turn_stats) {
    EffectQueue cur = {0}, post = {0};
    float reward = 0.0f;
    const float tile_reward = objective_tile_reward(env);
    const float combo_reward = objective_combo_bonus(env);
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
            add_goal_event(env, &stats, goal_score_slot(env), stats.candies);
            env->total_cleared += stats.candies;
            env->jelly_cleared += stats.events[goal_jelly_slot(env)];
            env->frosting_cleared += stats.events[goal_frosting_slot(env)];
            for (int i = 0; i < env->num_candies; i++) {
                env->color_collected[i] += stats.events[i];
            }
            apply_gravity(env);
            while ((dropped = drop_ingredients(env)) > 0) {
                add_goal_event(env, &stats, goal_ingredient_slot(env), dropped);
                apply_gravity(env);
            }
            env->ingredients_dropped += stats.events[goal_ingredient_slot(env)];
            reward += stats.candies * tile_reward;
            if (combo > 1) reward += (combo - 1) * combo_reward;
            if (turn_stats != NULL) merge_clear_stats(env, turn_stats, &stats);
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

static float resolve_special_swap(CandyCrush* env, int row, int col, int nrow, int ncol, unsigned char first, unsigned char second, ClearStats* turn_stats) {
    EffectQueue q = {0};
    const SpecialType sa = cell_special(first), sb = cell_special(second);
    const int other_row = row, other_col = col, moved_row = nrow, moved_col = ncol;
    if (sa == SPECIAL_COLOR_BOMB && sb == SPECIAL_COLOR_BOMB) {
        env->board[other_row][other_col] = 0; env->board[moved_row][moved_col] = 0;
        push_effect(&q, EFFECT_BOARD, moved_row, moved_col, 0, 0);
        return resolve_board(env, &q, false, 0, 0, 0, turn_stats);
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
        return resolve_board(env, &q, false, 0, 0, 0, turn_stats);
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
    return resolve_board(env, &q, false, 0, 0, 0, turn_stats);
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
    return write_action_mask(env, NULL);
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
    const int requested = env->goal_target[goal_ingredient_slot(env)];
    env->ingredient_total = 0;
    env->ingredient_remaining = 0;
    if (requested == 0) return;
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
        const int target = requested < 0 ? count : min_int(requested, count);
        for (int i = 0; i < target; i++) {
            const int pick = i + rng_int_bounded(env, count - i);
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
        if (env->goal_target[goal_jelly_slot(env)] != 0 && env->jelly_total == 0) add_jelly_rect(env, 2, 2, 2, 2);
        return;
    }
    for (int row = 0; row < env->board_size; row++) for (int col = 0; col < env->board_size; col++) {
        if (env->goal_target[goal_jelly_slot(env)] != 0 && rng_unit_float(env) < env->jelly_density) {
            env->jelly[row][col] = 1;
            env->jelly_total++;
            env->jelly_remaining++;
        }
        if (rng_unit_float(env) < env->frosting_density) env->frosting[row][col] = 1 + rng_int_bounded(env, max_int(1, env->frosting_layers));
    }
    if (env->goal_target[goal_jelly_slot(env)] != 0 && env->jelly_total == 0) {
        const int row = rng_int_bounded(env, env->board_size);
        const int col = rng_int_bounded(env, env->board_size);
        env->jelly[row][col] = 1;
        env->jelly_total = env->jelly_remaining = 1;
    }
    if (env->goal_target[goal_frosting_slot(env)] != 0 && total_frosting_layers(env) == 0) {
        const int row = rng_int_bounded(env, env->board_size);
        const int col = rng_int_bounded(env, env->board_size);
        env->frosting[row][col] = max_int(1, env->frosting_layers);
    }
}

static void resolve_auto_goals(CandyCrush* env) {
    if (env->goal_target[goal_jelly_slot(env)] < 0) env->goal_target[goal_jelly_slot(env)] = env->jelly_total;
    if (env->goal_target[goal_frosting_slot(env)] < 0) env->goal_target[goal_frosting_slot(env)] = total_frosting_layers(env);
    if (env->goal_target[goal_ingredient_slot(env)] < 0) env->goal_target[goal_ingredient_slot(env)] = env->ingredient_total;
    if (env->goal_target[goal_score_slot(env)] < 0) env->goal_target[goal_score_slot(env)] = max_int(1, env->max_steps * 3);
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
            const int row = rng_int_bounded(env, env->board_size);
            const int col = rng_int_bounded(env, env->board_size);
            const unsigned char cell = env->board[row][col];
            const int color = cell_color(cell);
            if (env->frosting[row][col] > 0 || is_empty(cell) || is_ingredient(cell)) continue;
            if (special == SPECIAL_COLOR_BOMB) env->board[row][col] = make_cell(0, SPECIAL_COLOR_BOMB);
            else if (special == SPECIAL_STRIPED_H) env->board[row][col] = make_cell(max_int(1, color), rng_int_bounded(env, 2) == 0 ? SPECIAL_STRIPED_H : SPECIAL_STRIPED_V);
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
    else if (env->task_distribution_mode != 0 && env->level_id < 0) sample_task_distribution(env);
    else restore_base_profile(env);
    env->steps = env->score = env->total_cleared = env->invalid_swaps = env->successful_swaps = 0;
    env->total_cascades = env->max_combo = env->reshuffles = 0;
    env->jelly_cleared = env->frosting_cleared = env->level_won = 0;
    env->ingredients_dropped = 0;
    memset(env->color_collected, 0, sizeof(env->color_collected));
    clear_goal_vector(env->goal_remaining);
    env->episode_return = 0.0f;
    clear_board_preserving_blockers(env);
    randomize_layout(env);
    generate_board(env);
    resolve_auto_goals(env);
    reset_goal_remaining(env);
    (void)update_observations(env);
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
    env->log.color_collected += total_color_collected(env);
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
    bool explicit_goal_vector = false;
    if (env->board_size < 4 || env->board_size > MAX_BOARD) { fprintf(stderr, "candy_crush: board_size must be in [4, %d]\n", MAX_BOARD); exit(1); }
    if (env->num_candies < 4 || env->num_candies > MAX_CANDIES) { fprintf(stderr, "candy_crush: num_candies must be in [4, %d]\n", MAX_CANDIES); exit(1); }
    if (env->max_steps < 1) { fprintf(stderr, "candy_crush: max_steps must be >= 1\n"); exit(1); }
    if (env->frosting_layers < 1) env->frosting_layers = 1;
    if (env->ingredient_spawn_rows < 1) env->ingredient_spawn_rows = 1;
    env->task_distribution_mode = env->task_distribution_mode != 0;
    env->task_min_active_goals = clamp_int(max_int(1, env->task_min_active_goals), 1, goal_slot_count(env));
    env->task_max_active_goals = clamp_int(max_int(env->task_min_active_goals, env->task_max_active_goals), env->task_min_active_goals, goal_slot_count(env));
    env->task_min_steps = max_int(1, env->task_min_steps);
    env->task_max_steps = max_int(env->task_min_steps, env->task_max_steps);
    if (env->progress_reward_scale < 0.0f) env->progress_reward_scale = 0.0f;
    if (env->shaping_gamma == 0.0f) env->shaping_gamma = 0.995f;
    env->shaping_gamma = clamp01(env->shaping_gamma);
    if (env->failure_penalty < 0.0f) env->failure_penalty = 0.0f;
    if (env->efficiency_bonus < 0.0f) env->efficiency_bonus = 0.0f;
    env->jelly_density = clamp01(env->jelly_density);
    env->frosting_density = clamp01(env->frosting_density);
    env->level_id = max_int(-1, env->level_id);
    env->curriculum_mode = env->curriculum_mode != 0;
    env->curriculum_start_level = clamp_int(env->curriculum_start_level, 0, MAX_LEVEL_BANK - 1);
    env->curriculum_max_level = clamp_int(env->curriculum_max_level, env->curriculum_start_level, MAX_LEVEL_BANK - 1);
    if (env->curriculum_min_episodes < 1) env->curriculum_min_episodes = 1;
    if (env->curriculum_min_episodes < 16) env->curriculum_min_episodes = 16;
    env->curriculum_threshold = clamp01(env->curriculum_threshold);
    if (env->curriculum_threshold < 0.40f) env->curriculum_threshold = 0.40f;
    env->curriculum_replay_prob = clamp01(env->curriculum_replay_prob);
    for (int i = 0; i < goal_slot_count(env); i++) {
        if (env->goal_target[i] != 0) explicit_goal_vector = true;
        if (env->goal_target[i] < GOAL_AUTO) env->goal_target[i] = 0;
    }
    env->has_goal_vector = env->has_goal_vector || explicit_goal_vector;
    if (!env->has_goal_vector && env->task_distribution_mode == 0 && env->level_id < 0 && env->curriculum_mode == 0) {
        fprintf(stderr, "candy_crush: provide goal_vector or enable task_distribution/curriculum/level profiles\n");
        exit(1);
    }
    clear_goal_vector(env->goal_remaining);
    env->base_max_steps = env->max_steps;
    env->base_frosting_layers = env->frosting_layers;
    env->base_ingredient_spawn_rows = env->ingredient_spawn_rows;
    copy_goal_vector(env->base_goal_target, env->goal_target);
    env->base_jelly_density = env->jelly_density;
    env->base_frosting_density = env->frosting_density;
    env->active_level = env->level_id >= 0 ? clamp_int(env->level_id, env->curriculum_start_level, env->curriculum_max_level) : -1;
    env->unlocked_level = env->active_level >= 0 ? env->active_level : env->curriculum_start_level;
    env->frontier_level = env->active_level >= 0 ? env->active_level : env->curriculum_start_level;
    env->frontier_episodes = 0;
    env->frontier_wins = 0;
    ensure_rng_seeded(env);
    memset(&env->log, 0, sizeof(Log));
}

static void c_reset(CandyCrush* env) {
    if (env->terminals) env->terminals[0] = 0;
    if (env->rewards) env->rewards[0] = 0.0f;
    ensure_rng_seeded(env);
    reset_episode(env);
}

static void c_step(CandyCrush* env) {
    int row, col, nrow, ncol, dir;
    ClearStats turn_stats = {0};
    const float phi_before = goal_potential(env);
    bool observations_ready = false;
    float reward = 0.0f;
    env->rewards[0] = 0.0f; env->terminals[0] = 0; env->steps++;
    if (!decode_action(env, &row, &col, &nrow, &ncol, &dir) || !swappable_cell(env, row, col) || !swappable_cell(env, nrow, ncol)) {
        reward = env->invalid_penalty; env->invalid_swaps++;
    } else {
        const unsigned char first = env->board[row][col], second = env->board[nrow][ncol];
        swap_cells(env, row, col, nrow, ncol);
        if (auto_swap(first, second)) { env->successful_swaps++; reward = resolve_special_swap(env, row, col, nrow, ncol, first, second, &turn_stats); }
        else if (!swap_creates_match(env, row, col, nrow, ncol)) { swap_cells(env, row, col, nrow, ncol); reward = env->invalid_penalty; env->invalid_swaps++; }
        else { env->successful_swaps++; reward = resolve_board(env, NULL, true, nrow, ncol, dir, &turn_stats); }
        apply_goal_events(env, &turn_stats);
    }
    if (!goal_complete(env)) {
        observations_ready = true;
        if (!update_observations(env)) {
            reshuffle_board(env);
            env->reshuffles++;
            reward += env->shuffle_penalty;
            (void)update_observations(env);
        }
    }
    {
        const float phi_after = goal_potential(env);
        reward += env->progress_reward_scale * (env->shaping_gamma * phi_after - phi_before);
    }
    if (goal_complete(env)) {
        reward += env->success_bonus
            + env->efficiency_bonus * ((float)max_int(0, env->max_steps - env->steps) / max_int(1, env->max_steps));
        env->level_won = 1;
        env->episode_return += reward;
        env->rewards[0] = reward;
        env->terminals[0] = 1;
        record_curriculum_result(env, true);
        write_episode_log(env);
        reset_episode(env);
        env->rewards[0] = reward;
        return;
    }
    if (env->steps >= env->max_steps) {
        reward -= env->failure_penalty;
        env->episode_return += reward;
        env->rewards[0] = reward;
        env->terminals[0] = 1; record_curriculum_result(env, false); write_episode_log(env); reset_episode(env); env->rewards[0] = reward;
        return;
    }
    env->episode_return += reward; env->rewards[0] = reward;
    if (!observations_ready) (void)update_observations(env);
}

static inline char special_marker(unsigned char cell) {
    if (cell_special(cell) == SPECIAL_STRIPED_H) return '-';
    if (cell_special(cell) == SPECIAL_STRIPED_V) return '|';
    if (cell_special(cell) == SPECIAL_WRAPPED) return 'W';
    if (cell_special(cell) == SPECIAL_COLOR_BOMB) return 'O';
    if (cell_special(cell) == SPECIAL_FISH) return 'F';
    return ' ';
}

static inline const char* candy_name(CandyCrush* env, int color) {
    static const char* names[MAX_CANDIES + 1] = {
        "None", "Red", "Green", "Blue", "Yellow", "Purple", "Teal", "Orange", "White"
    };
    return names[clamp_int(color, 0, MAX_CANDIES)];
}

static inline const char* goal_slot_name(CandyCrush* env, int slot) {
    if (slot < env->num_candies) return candy_name(env, slot + 1);
    if (slot == goal_jelly_slot(env)) return "Jelly";
    if (slot == goal_frosting_slot(env)) return "Frost";
    if (slot == goal_ingredient_slot(env)) return "Ingredient";
    return "Score";
}

static void format_goal_label(CandyCrush* env, char* label, size_t label_size, bool remaining) {
    int written = 0;
    bool first = true;
    label[0] = '\0';
    for (int i = 0; i < goal_slot_count(env); i++) {
        const int value = remaining ? env->goal_remaining[i] : max_int(0, env->goal_target[i]);
        if (value <= 0) continue;
        written += snprintf(
            label + written,
            written < (int)label_size ? label_size - written : 0,
            "%s%s %d",
            first ? "" : ", ",
            goal_slot_name(env, i),
            value
        );
        first = false;
        if (written >= (int)label_size - 1) break;
    }
    if (first) snprintf(label, label_size, "None");
}

static void c_render(CandyCrush* env) {
    const int cell = 64, gap = 6, width = env->board_size * cell, height = env->board_size * cell + 136;
    char label[192];
    char goal_label[192];
    char remaining_label[192];
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
    format_goal_label(env, goal_label, sizeof(goal_label), false);
    format_goal_label(env, remaining_label, sizeof(remaining_label), true);
    snprintf(label, sizeof(label), "Level: %d  Unlocked: %d  Goal: %s", env->active_level >= 0 ? env->active_level : 0, env->unlocked_level, goal_label);
    DrawText(label, 12, env->board_size * cell + 8, 22, PUFF_WHITE);
    snprintf(label, sizeof(label), "Score: %d  Steps: %d/%d", env->score, env->steps, env->max_steps); DrawText(label, 12, env->board_size * cell + 34, 22, PUFF_WHITE);
    snprintf(label, sizeof(label), "Remaining: %s", remaining_label);
    DrawText(label, 12, env->board_size * cell + 60, 20, PUFF_WHITE);
    snprintf(label, sizeof(label), "Ingredients: %d/%d  Combo: %d", env->ingredients_dropped, env->ingredient_total, env->max_combo); DrawText(label, 12, env->board_size * cell + 84, 20, PUFF_WHITE);
    snprintf(label, sizeof(label), "Goal remaining: %.2f  Curriculum WR: %.2f", goal_remaining_ratio(env), curriculum_win_rate(env)); DrawText(label, 12, env->board_size * cell + 104, 20, PUFF_WHITE);
    EndDrawing();
}

static void c_close(CandyCrush* env) { if (IsWindowReady()) CloseWindow(); }

#endif
