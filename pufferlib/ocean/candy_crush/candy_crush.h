#ifndef PUFFERLIB_CANDY_CRUSH_H
#define PUFFERLIB_CANDY_CRUSH_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define MAX_BOARD 10
#define MAX_CANDIES 8

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float total_cleared;
    float invalid_swaps;
    float successful_swaps;
    float total_cascades;
    float max_combo;
    float reshuffles;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int board_size;
    int num_candies;
    int max_steps;

    float reward_per_tile;
    float combo_bonus;
    float invalid_penalty;
    float shuffle_penalty;

    int steps;
    int score;
    int total_cleared;
    int invalid_swaps;
    int successful_swaps;
    int total_cascades;
    int max_combo;
    int reshuffles;
    float episode_return;

    unsigned char board[MAX_BOARD][MAX_BOARD];
} CandyCrush;

static const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 255};
static const Color GRID_COLOR = (Color){24, 64, 64, 255};
static const Color CANDY_COLORS[MAX_CANDIES + 1] = {
    {20, 20, 20, 255},
    {231, 76, 60, 255},
    {46, 204, 113, 255},
    {52, 152, 219, 255},
    {241, 196, 15, 255},
    {155, 89, 182, 255},
    {26, 188, 156, 255},
    {230, 126, 34, 255},
    {236, 240, 241, 255},
};

static const char CANDY_SYMBOLS[MAX_CANDIES + 1] = {
    '.',
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
};

static inline int max_int(int a, int b) { return a > b ? a : b; }

static inline unsigned char sample_candy(CandyCrush* env) {
    return (unsigned char)(1 + rand() % env->num_candies);
}

static void update_observations(CandyCrush* env) {
    const int cells = env->board_size * env->board_size;
    const int obs_size = cells * (env->num_candies + 1);
    memset(env->observations, 0, obs_size * sizeof(unsigned char));

    for (int row = 0; row < env->board_size; row++) {
        for (int col = 0; col < env->board_size; col++) {
            const unsigned char candy = env->board[row][col];
            const int board_idx = row * env->board_size + col;
            if (candy > 0 && candy <= env->num_candies) {
                env->observations[(candy - 1) * cells + board_idx] = 1;
            }
        }
    }

    const int remaining = max_int(0, env->max_steps - env->steps);
    const unsigned char remaining_scaled = (unsigned char)(
        255 * remaining / max_int(1, env->max_steps)
    );
    for (int idx = 0; idx < cells; idx++) {
        env->observations[env->num_candies * cells + idx] = remaining_scaled;
    }
}

static int find_matches(CandyCrush* env, bool marked[MAX_BOARD][MAX_BOARD]) {
    int total = 0;
    memset(marked, 0, sizeof(bool) * MAX_BOARD * MAX_BOARD);

    for (int row = 0; row < env->board_size; row++) {
        int start = 0;
        while (start < env->board_size) {
            const unsigned char candy = env->board[row][start];
            int end = start + 1;
            while (end < env->board_size && env->board[row][end] == candy) {
                end++;
            }
            if (candy != 0 && end - start >= 3) {
                for (int col = start; col < end; col++) {
                    if (!marked[row][col]) {
                        marked[row][col] = true;
                        total++;
                    }
                }
            }
            start = end;
        }
    }

    for (int col = 0; col < env->board_size; col++) {
        int start = 0;
        while (start < env->board_size) {
            const unsigned char candy = env->board[start][col];
            int end = start + 1;
            while (end < env->board_size && env->board[end][col] == candy) {
                end++;
            }
            if (candy != 0 && end - start >= 3) {
                for (int row = start; row < end; row++) {
                    if (!marked[row][col]) {
                        marked[row][col] = true;
                        total++;
                    }
                }
            }
            start = end;
        }
    }

    return total;
}

static void clear_matches(CandyCrush* env, bool marked[MAX_BOARD][MAX_BOARD]) {
    for (int row = 0; row < env->board_size; row++) {
        for (int col = 0; col < env->board_size; col++) {
            if (marked[row][col]) {
                env->board[row][col] = 0;
            }
        }
    }
}

static void apply_gravity(CandyCrush* env) {
    for (int col = 0; col < env->board_size; col++) {
        int write_row = env->board_size - 1;
        for (int row = env->board_size - 1; row >= 0; row--) {
            if (env->board[row][col] != 0) {
                env->board[write_row][col] = env->board[row][col];
                write_row--;
            }
        }
        while (write_row >= 0) {
            env->board[write_row][col] = 0;
            write_row--;
        }
    }
}

static void refill_board(CandyCrush* env) {
    for (int col = 0; col < env->board_size; col++) {
        for (int row = 0; row < env->board_size; row++) {
            if (env->board[row][col] == 0) {
                env->board[row][col] = sample_candy(env);
            }
        }
    }
}

static bool has_legal_moves(CandyCrush* env) {
    bool marked[MAX_BOARD][MAX_BOARD];
    for (int row = 0; row < env->board_size; row++) {
        for (int col = 0; col < env->board_size; col++) {
            const int drows[2] = {0, 1};
            const int dcols[2] = {1, 0};
            for (int i = 0; i < 2; i++) {
                const int nrow = row + drows[i];
                const int ncol = col + dcols[i];
                if (nrow >= env->board_size || ncol >= env->board_size) {
                    continue;
                }

                const unsigned char tmp = env->board[row][col];
                env->board[row][col] = env->board[nrow][ncol];
                env->board[nrow][ncol] = tmp;

                const bool legal = find_matches(env, marked) > 0;

                env->board[nrow][ncol] = env->board[row][col];
                env->board[row][col] = tmp;

                if (legal) {
                    return true;
                }
            }
        }
    }

    return false;
}

static void generate_board(CandyCrush* env) {
    for (int attempt = 0; attempt < 256; attempt++) {
        for (int row = 0; row < env->board_size; row++) {
            for (int col = 0; col < env->board_size; col++) {
                unsigned char candy = sample_candy(env);
                for (int retry = 0; retry < 32; retry++) {
                    const bool row_match = col >= 2
                        && env->board[row][col - 1] == candy
                        && env->board[row][col - 2] == candy;
                    const bool col_match = row >= 2
                        && env->board[row - 1][col] == candy
                        && env->board[row - 2][col] == candy;
                    if (!row_match && !col_match) {
                        break;
                    }
                    candy = sample_candy(env);
                }
                env->board[row][col] = candy;
            }
        }

        if (has_legal_moves(env)) {
            return;
        }
    }
}

static void reset_episode(CandyCrush* env) {
    env->steps = 0;
    env->score = 0;
    env->total_cleared = 0;
    env->invalid_swaps = 0;
    env->successful_swaps = 0;
    env->total_cascades = 0;
    env->max_combo = 0;
    env->reshuffles = 0;
    env->episode_return = 0.0f;

    memset(env->board, 0, sizeof(env->board));
    generate_board(env);
    update_observations(env);
}

static void write_episode_log(CandyCrush* env) {
    env->log.score += (float)env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.total_cleared += (float)env->total_cleared;
    env->log.invalid_swaps += (float)env->invalid_swaps;
    env->log.successful_swaps += (float)env->successful_swaps;
    env->log.total_cascades += (float)env->total_cascades;
    env->log.max_combo += (float)env->max_combo;
    env->log.reshuffles += (float)env->reshuffles;
    env->log.n += 1.0f;
}

static float resolve_matches(CandyCrush* env) {
    bool marked[MAX_BOARD][MAX_BOARD];
    int matched = find_matches(env, marked);
    float reward = 0.0f;
    int combo = 0;

    while (matched > 0) {
        combo++;
        env->score += matched;
        env->total_cleared += matched;
        reward += matched * env->reward_per_tile;
        if (combo > 1) {
            reward += (combo - 1) * env->combo_bonus;
        }

        clear_matches(env, marked);
        apply_gravity(env);
        refill_board(env);
        matched = find_matches(env, marked);
    }

    env->max_combo = max_int(env->max_combo, combo);
    env->total_cascades += max_int(0, combo - 1);
    return reward;
}

static inline bool decode_action(
    CandyCrush* env,
    int* row,
    int* col,
    int* nrow,
    int* ncol
) {
    const int action_count = env->board_size * env->board_size * 4;
    int action = env->actions[0];
    if (action < 0) {
        action = 0;
    }
    action %= action_count;

    const int direction = action % 4;
    const int cell = action / 4;
    *row = cell / env->board_size;
    *col = cell % env->board_size;
    *nrow = *row;
    *ncol = *col;

    if (direction == 0) {
        *nrow -= 1;
    } else if (direction == 1) {
        *ncol += 1;
    } else if (direction == 2) {
        *nrow += 1;
    } else {
        *ncol -= 1;
    }

    return *nrow >= 0 && *nrow < env->board_size && *ncol >= 0 && *ncol < env->board_size;
}

static inline void swap_cells(CandyCrush* env, int row, int col, int nrow, int ncol) {
    const unsigned char tmp = env->board[row][col];
    env->board[row][col] = env->board[nrow][ncol];
    env->board[nrow][ncol] = tmp;
}

static void init_env(CandyCrush* env) {
    if (env->board_size < 4 || env->board_size > MAX_BOARD) {
        fprintf(stderr, "candy_crush: board_size must be in [4, %d]\n", MAX_BOARD);
        exit(1);
    }
    if (env->num_candies < 4 || env->num_candies > MAX_CANDIES) {
        fprintf(stderr, "candy_crush: num_candies must be in [4, %d]\n", MAX_CANDIES);
        exit(1);
    }
    if (env->max_steps < 1) {
        fprintf(stderr, "candy_crush: max_steps must be >= 1\n");
        exit(1);
    }
    memset(&env->log, 0, sizeof(Log));
}

static void c_reset(CandyCrush* env) {
    if (env->terminals) {
        env->terminals[0] = 0;
    }
    if (env->rewards) {
        env->rewards[0] = 0.0f;
    }
    reset_episode(env);
}

static void c_step(CandyCrush* env) {
    int row, col, nrow, ncol;
    bool marked[MAX_BOARD][MAX_BOARD];
    float reward = 0.0f;

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->steps += 1;

    if (!decode_action(env, &row, &col, &nrow, &ncol)) {
        reward = env->invalid_penalty;
        env->invalid_swaps += 1;
    } else {
        swap_cells(env, row, col, nrow, ncol);
        if (find_matches(env, marked) == 0) {
            swap_cells(env, row, col, nrow, ncol);
            reward = env->invalid_penalty;
            env->invalid_swaps += 1;
        } else {
            env->successful_swaps += 1;
            reward = resolve_matches(env);
            if (!has_legal_moves(env)) {
                generate_board(env);
                env->reshuffles += 1;
                reward += env->shuffle_penalty;
            }
        }
    }

    env->episode_return += reward;
    env->rewards[0] = reward;
    update_observations(env);

    if (env->steps >= env->max_steps) {
        env->terminals[0] = 1;
        write_episode_log(env);
        reset_episode(env);
        env->rewards[0] = reward;
    }
}

static void c_render(CandyCrush* env) {
    const int cell = 64;
    const int gap = 6;
    const int width = env->board_size * cell;
    const int height = env->board_size * cell + 80;
    char label[64];

    if (!IsWindowReady()) {
        InitWindow(width, height, "PufferLib Candy Crush");
        SetTargetFPS(30);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    for (int row = 0; row < env->board_size; row++) {
        for (int col = 0; col < env->board_size; col++) {
            const unsigned char candy = env->board[row][col];
            const int color_idx = candy <= MAX_CANDIES ? candy : 0;
            const int x = col * cell + gap / 2;
            const int y = row * cell + gap / 2;

            DrawRectangle(x, y, cell - gap, cell - gap, GRID_COLOR);
            DrawCircle(x + (cell - gap) / 2, y + (cell - gap) / 2, 22, CANDY_COLORS[color_idx]);

            snprintf(label, sizeof(label), "%c", CANDY_SYMBOLS[color_idx]);
            DrawText(label, x + 24, y + 18, 20, PUFF_WHITE);
        }
    }

    snprintf(label, sizeof(label), "Score: %d", env->score);
    DrawText(label, 12, env->board_size * cell + 12, 24, PUFF_WHITE);
    snprintf(label, sizeof(label), "Steps: %d/%d", env->steps, env->max_steps);
    DrawText(label, 180, env->board_size * cell + 12, 24, PUFF_WHITE);
    snprintf(label, sizeof(label), "Combo: %d", env->max_combo);
    DrawText(label, 380, env->board_size * cell + 12, 24, PUFF_WHITE);

    EndDrawing();
}

static void c_close(CandyCrush* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif

