#ifndef PUFFERLIB_BLOCK_PUZZLE_H
#define PUFFERLIB_BLOCK_PUZZLE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define BOARD_SIZE 10
#define MAX_BOARD_SIZE BOARD_SIZE
#define PREVIEW_SIZE 5
#define HAND_SIZE 3
#define ACTION_ROTATIONS 4
#define MAX_PIECE_TYPES 15
#define MAX_PIECE_CELLS 9

#define OBS_BOARD_CELLS (BOARD_SIZE * BOARD_SIZE)
#define OBS_PREVIEW_CELLS (HAND_SIZE * PREVIEW_SIZE * PREVIEW_SIZE)
#define OBS_SCALAR_CELLS HAND_SIZE
#define ACTION_COUNT (HAND_SIZE * BOARD_SIZE * BOARD_SIZE * ACTION_ROTATIONS)
#define OBS_TOTAL_SIZE (OBS_BOARD_CELLS + OBS_PREVIEW_CELLS + OBS_SCALAR_CELLS + ACTION_COUNT)

#define BIT_AT(row, col) (1u << ((row) * PREVIEW_SIZE + (col)))

typedef struct Log {
    float score;
    float lines_cleared;
    float pieces_placed;
    float invalid_actions;
    float board_fill;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct PieceRotation {
    unsigned char width;
    unsigned char height;
    unsigned char cells;
    signed char rows[MAX_PIECE_CELLS];
    signed char cols[MAX_PIECE_CELLS];
    unsigned int mask;
} PieceRotation;

typedef struct PieceDef {
    unsigned char rotation_count;
    unsigned char cells;
    unsigned int preview_mask;
    PieceRotation rotations[ACTION_ROTATIONS];
} PieceDef;

typedef struct BlockPuzzle {
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    Log log;

    int board_size;
    int allow_rotations;
    float reward_per_block;
    float line_bonus;
    float multi_line_bonus;
    float invalid_penalty;
    float loss_penalty;

    int score;
    int steps;
    int lines_cleared;
    int pieces_placed;
    int invalid_actions;
    int board_fill_peak;
    float episode_return;

    unsigned int rng;
    unsigned char board[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    unsigned char hand_piece[HAND_SIZE];
    unsigned char hand_active[HAND_SIZE];
} BlockPuzzle;

static PieceDef BLOCK_PIECES[MAX_PIECE_TYPES];
static int BLOCK_PIECES_READY = 0;
static bool BLOCK_PUZZLE_WINDOW_READY = false;

static const unsigned int BLOCK_PIECE_SEEDS[MAX_PIECE_TYPES] = {
    BIT_AT(0, 0),
    BIT_AT(0, 0) | BIT_AT(0, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(1, 0),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(1, 0) | BIT_AT(1, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2) | BIT_AT(0, 3),
    BIT_AT(0, 0) | BIT_AT(1, 0) | BIT_AT(2, 0) | BIT_AT(2, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2) | BIT_AT(1, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(1, 1) | BIT_AT(1, 2),
    BIT_AT(0, 1) | BIT_AT(0, 2) | BIT_AT(1, 0) | BIT_AT(1, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2) | BIT_AT(0, 3) | BIT_AT(0, 4),
    BIT_AT(0, 0) | BIT_AT(1, 0) | BIT_AT(2, 0) | BIT_AT(3, 0) | BIT_AT(3, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2) | BIT_AT(1, 1) | BIT_AT(2, 1),
    BIT_AT(0, 1) | BIT_AT(1, 0) | BIT_AT(1, 1) | BIT_AT(1, 2) | BIT_AT(2, 1),
    BIT_AT(0, 0) | BIT_AT(0, 1) | BIT_AT(0, 2) |
        BIT_AT(1, 0) | BIT_AT(1, 1) | BIT_AT(1, 2) |
        BIT_AT(2, 0) | BIT_AT(2, 1) | BIT_AT(2, 2),
};

static const Color PUFF_BG = (Color){6, 24, 24, 255};
static const Color PUFF_GRID = (Color){24, 64, 64, 255};
static const Color PUFF_FILL = (Color){0, 187, 187, 255};
static const Color PUFF_TEXT = (Color){241, 241, 241, 255};
static const Color PUFF_GHOST = (Color){72, 108, 108, 255};
static const Color PUFF_USED = (Color){160, 80, 80, 255};

static inline int max_int(int a, int b) { return a > b ? a : b; }
static inline int min_int(int a, int b) { return a < b ? a : b; }

static void init_env(BlockPuzzle* env);
static void c_reset(BlockPuzzle* env);
static void c_step(BlockPuzzle* env);
static void c_render(BlockPuzzle* env);
static void c_close(BlockPuzzle* env);
static void allocate(BlockPuzzle* env);
static void free_allocated(BlockPuzzle* env);

static inline int rng_int_bounded(BlockPuzzle* env, int upper) {
    return upper <= 1 ? 0 : (int)(rand_r(&env->rng) % (unsigned int)upper);
}

static unsigned int rotate_mask90(unsigned int mask) {
    unsigned int rotated = 0;
    int row;
    int col;

    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            if ((mask & BIT_AT(row, col)) == 0) {
                continue;
            }
            rotated |= BIT_AT(col, PREVIEW_SIZE - 1 - row);
        }
    }
    return rotated;
}

static unsigned int normalize_mask(unsigned int mask) {
    int min_row = PREVIEW_SIZE;
    int min_col = PREVIEW_SIZE;
    int row;
    int col;
    unsigned int normalized = 0;

    if (mask == 0) {
        return 0;
    }

    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            if ((mask & BIT_AT(row, col)) == 0) {
                continue;
            }
            min_row = min_int(min_row, row);
            min_col = min_int(min_col, col);
        }
    }

    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            if ((mask & BIT_AT(row, col)) == 0) {
                continue;
            }
            normalized |= BIT_AT(row - min_row, col - min_col);
        }
    }
    return normalized;
}

static void fill_rotation(PieceRotation* rotation, unsigned int mask) {
    int min_row = PREVIEW_SIZE;
    int min_col = PREVIEW_SIZE;
    int max_row = 0;
    int max_col = 0;
    int cells = 0;
    int row;
    int col;

    memset(rotation, 0, sizeof(PieceRotation));
    rotation->mask = mask;
    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            if ((mask & BIT_AT(row, col)) == 0) {
                continue;
            }
            min_row = min_int(min_row, row);
            min_col = min_int(min_col, col);
            max_row = max_int(max_row, row);
            max_col = max_int(max_col, col);
        }
    }

    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            if ((mask & BIT_AT(row, col)) == 0) {
                continue;
            }
            rotation->rows[cells] = (signed char)(row - min_row);
            rotation->cols[cells] = (signed char)(col - min_col);
            cells++;
        }
    }

    rotation->cells = (unsigned char)cells;
    rotation->height = (unsigned char)(max_row - min_row + 1);
    rotation->width = (unsigned char)(max_col - min_col + 1);
}

static void build_piece_bank(void) {
    int piece_count;
    int piece_idx;

    if (BLOCK_PIECES_READY) {
        return;
    }

    piece_count = (int)(sizeof(BLOCK_PIECE_SEEDS) / sizeof(BLOCK_PIECE_SEEDS[0]));
    for (piece_idx = 0; piece_idx < piece_count; piece_idx++) {
        PieceDef* piece = &BLOCK_PIECES[piece_idx];
        unsigned int mask = normalize_mask(BLOCK_PIECE_SEEDS[piece_idx]);
        int rot;

        memset(piece, 0, sizeof(PieceDef));
        piece->preview_mask = mask;

        for (rot = 0; rot < ACTION_ROTATIONS; rot++) {
            bool duplicate = false;
            int existing;
            for (existing = 0; existing < piece->rotation_count; existing++) {
                if (piece->rotations[existing].mask == mask) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                fill_rotation(&piece->rotations[piece->rotation_count], mask);
                piece->rotation_count++;
            }
            mask = normalize_mask(rotate_mask90(mask));
        }

        piece->cells = piece->rotations[0].cells;
    }

    BLOCK_PIECES_READY = 1;
}

static inline int board_fill(BlockPuzzle* env) {
    int total = 0;
    int row;
    int col;

    for (row = 0; row < env->board_size; row++) {
        for (col = 0; col < env->board_size; col++) {
            total += env->board[row][col] != 0;
        }
    }
    return total;
}

static inline bool any_active_piece(BlockPuzzle* env) {
    int slot;
    for (slot = 0; slot < HAND_SIZE; slot++) {
        if (env->hand_active[slot]) {
            return true;
        }
    }
    return false;
}

static inline void clear_board(BlockPuzzle* env) {
    memset(env->board, 0, sizeof(env->board));
}

static void draw_hand(BlockPuzzle* env) {
    const int piece_count = (int)(sizeof(BLOCK_PIECE_SEEDS) / sizeof(BLOCK_PIECE_SEEDS[0]));
    int slot;

    for (slot = 0; slot < HAND_SIZE; slot++) {
        env->hand_piece[slot] = (unsigned char)rng_int_bounded(env, piece_count);
        env->hand_active[slot] = 1;
    }
}

static bool can_place(BlockPuzzle* env, int slot, int row, int col, int rotation_idx) {
    PieceDef* piece;
    PieceRotation* rotation;
    int i;

    if (slot < 0 || slot >= HAND_SIZE) {
        return false;
    }
    if (!env->hand_active[slot]) {
        return false;
    }

    piece = &BLOCK_PIECES[env->hand_piece[slot]];
    if (rotation_idx < 0 || rotation_idx >= piece->rotation_count) {
        return false;
    }
    if (!env->allow_rotations && rotation_idx != 0) {
        return false;
    }

    rotation = &piece->rotations[rotation_idx];
    if (row < 0 || col < 0) {
        return false;
    }
    if (row + rotation->height > env->board_size) {
        return false;
    }
    if (col + rotation->width > env->board_size) {
        return false;
    }

    for (i = 0; i < rotation->cells; i++) {
        const int rr = row + rotation->rows[i];
        const int cc = col + rotation->cols[i];
        if (env->board[rr][cc]) {
            return false;
        }
    }

    return true;
}

static bool write_action_mask(BlockPuzzle* env, unsigned char* action_mask) {
    const int slot_stride = BOARD_SIZE * BOARD_SIZE * ACTION_ROTATIONS;
    bool any_legal = false;
    int slot;
    int row;
    int col;
    int rotation_idx;

    memset(action_mask, 0, ACTION_COUNT);
    for (slot = 0; slot < HAND_SIZE; slot++) {
        for (row = 0; row < BOARD_SIZE; row++) {
            for (col = 0; col < BOARD_SIZE; col++) {
                const int anchor = (row * BOARD_SIZE + col) * ACTION_ROTATIONS;
                for (rotation_idx = 0; rotation_idx < ACTION_ROTATIONS; rotation_idx++) {
                    const unsigned char legal = can_place(env, slot, row, col, rotation_idx) ? 1 : 0;
                    action_mask[slot * slot_stride + anchor + rotation_idx] = legal;
                    any_legal = any_legal || legal != 0;
                }
            }
        }
    }
    return any_legal;
}

static void write_board_obs(BlockPuzzle* env, unsigned char* board_obs) {
    int row;
    int col;

    memset(board_obs, 0, OBS_BOARD_CELLS);
    for (row = 0; row < env->board_size; row++) {
        for (col = 0; col < env->board_size; col++) {
            board_obs[row * BOARD_SIZE + col] = env->board[row][col] ? 1 : 0;
        }
    }
}

static void write_preview_obs(BlockPuzzle* env, unsigned char* preview_obs) {
    int slot;
    int row;
    int col;

    memset(preview_obs, 0, OBS_PREVIEW_CELLS);
    for (slot = 0; slot < HAND_SIZE; slot++) {
        unsigned char* dst;
        PieceDef* piece;

        if (!env->hand_active[slot]) {
            continue;
        }

        piece = &BLOCK_PIECES[env->hand_piece[slot]];
        dst = preview_obs + slot * PREVIEW_SIZE * PREVIEW_SIZE;
        for (row = 0; row < PREVIEW_SIZE; row++) {
            for (col = 0; col < PREVIEW_SIZE; col++) {
                if (piece->preview_mask & BIT_AT(row, col)) {
                    dst[row * PREVIEW_SIZE + col] = 1;
                }
            }
        }
    }
}

static void write_scalar_obs(BlockPuzzle* env, unsigned char* scalar_obs) {
    int slot;
    for (slot = 0; slot < HAND_SIZE; slot++) {
        scalar_obs[slot] = env->hand_active[slot] ? 1 : 0;
    }
}

static bool update_observations(BlockPuzzle* env) {
    unsigned char* board_obs = env->observations;
    unsigned char* preview_obs = board_obs + OBS_BOARD_CELLS;
    unsigned char* scalar_obs = preview_obs + OBS_PREVIEW_CELLS;
    unsigned char* action_mask = scalar_obs + OBS_SCALAR_CELLS;

    memset(env->observations, 0, OBS_TOTAL_SIZE);
    write_board_obs(env, board_obs);
    write_preview_obs(env, preview_obs);
    write_scalar_obs(env, scalar_obs);
    return write_action_mask(env, action_mask);
}

static void place_piece(BlockPuzzle* env, int slot, int row, int col, int rotation_idx) {
    PieceRotation* rotation = &BLOCK_PIECES[env->hand_piece[slot]].rotations[rotation_idx];
    int i;

    for (i = 0; i < rotation->cells; i++) {
        env->board[row + rotation->rows[i]][col + rotation->cols[i]] = 1;
    }
    env->hand_active[slot] = 0;
}

static int clear_completed_lines(BlockPuzzle* env, int* cleared_lines) {
    bool full_rows[BOARD_SIZE] = {0};
    bool full_cols[BOARD_SIZE] = {0};
    int lines = 0;
    int cleared_cells = 0;
    int row;
    int col;

    for (row = 0; row < env->board_size; row++) {
        bool full = true;
        for (col = 0; col < env->board_size; col++) {
            if (!env->board[row][col]) {
                full = false;
                break;
            }
        }
        if (full) {
            full_rows[row] = true;
            lines++;
        }
    }

    for (col = 0; col < env->board_size; col++) {
        bool full = true;
        for (row = 0; row < env->board_size; row++) {
            if (!env->board[row][col]) {
                full = false;
                break;
            }
        }
        if (full) {
            full_cols[col] = true;
            lines++;
        }
    }

    for (row = 0; row < env->board_size; row++) {
        for (col = 0; col < env->board_size; col++) {
            if (!(full_rows[row] || full_cols[col])) {
                continue;
            }
            if (env->board[row][col]) {
                cleared_cells++;
            }
            env->board[row][col] = 0;
        }
    }

    *cleared_lines = lines;
    return cleared_cells;
}

static void write_episode_log(BlockPuzzle* env) {
    env->log.score += (float)env->score;
    env->log.lines_cleared += (float)env->lines_cleared;
    env->log.pieces_placed += (float)env->pieces_placed;
    env->log.invalid_actions += (float)env->invalid_actions;
    env->log.board_fill += (float)env->board_fill_peak;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.n += 1.0f;
}

static void start_episode(BlockPuzzle* env) {
    clear_board(env);
    draw_hand(env);
    env->score = 0;
    env->steps = 0;
    env->lines_cleared = 0;
    env->pieces_placed = 0;
    env->invalid_actions = 0;
    env->board_fill_peak = 0;
    env->episode_return = 0.0f;
    update_observations(env);
}

static void init_env(BlockPuzzle* env) {
    build_piece_bank();

    env->board_size = BOARD_SIZE;
    env->allow_rotations = env->allow_rotations ? 1 : 0;
    if (env->reward_per_block <= 0.0f) {
        env->reward_per_block = 0.10f;
    }
    if (env->line_bonus <= 0.0f) {
        env->line_bonus = 1.0f;
    }
    if (env->multi_line_bonus < 0.0f) {
        env->multi_line_bonus = 0.0f;
    }
    if (env->invalid_penalty >= 0.0f) {
        env->invalid_penalty = -0.25f;
    }
    if (env->loss_penalty >= 0.0f) {
        env->loss_penalty = -1.0f;
    }
}

static void allocate(BlockPuzzle* env) {
    init_env(env);
    env->observations = (unsigned char*)calloc(OBS_TOTAL_SIZE, sizeof(unsigned char));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

static void free_allocated(BlockPuzzle* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    env->observations = NULL;
    env->actions = NULL;
    env->rewards = NULL;
    env->terminals = NULL;
    c_close(env);
}

static void c_reset(BlockPuzzle* env) {
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;
    start_episode(env);
}

static void c_step(BlockPuzzle* env) {
    const int slot_stride = BOARD_SIZE * BOARD_SIZE * ACTION_ROTATIONS;
    float reward = 0.0f;
    int action = (int)env->actions[0];

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->steps++;

    if (action < 0 || action >= ACTION_COUNT) {
        env->invalid_actions++;
        reward = env->invalid_penalty;
        update_observations(env);
        env->rewards[0] = reward;
        env->episode_return += reward;
        return;
    }

    {
        const int slot = action / slot_stride;
        const int rem = action % slot_stride;
        const int cell = rem / ACTION_ROTATIONS;
        const int rotation_idx = rem % ACTION_ROTATIONS;
        const int row = cell / BOARD_SIZE;
        const int col = cell % BOARD_SIZE;
        PieceRotation* rotation;
        int cleared_lines = 0;
        int cleared_cells;
        bool any_legal;

        if (!can_place(env, slot, row, col, rotation_idx)) {
            env->invalid_actions++;
            reward = env->invalid_penalty;
            update_observations(env);
            env->rewards[0] = reward;
            env->episode_return += reward;
            return;
        }

        rotation = &BLOCK_PIECES[env->hand_piece[slot]].rotations[rotation_idx];
        place_piece(env, slot, row, col, rotation_idx);
        env->pieces_placed++;
        env->board_fill_peak = max_int(env->board_fill_peak, board_fill(env));

        cleared_cells = clear_completed_lines(env, &cleared_lines);
        env->lines_cleared += cleared_lines;
        env->score += rotation->cells + cleared_cells;

        reward += env->reward_per_block * (float)rotation->cells;
        if (cleared_lines > 0) {
            reward += env->line_bonus * (float)cleared_lines;
            reward += env->multi_line_bonus * (float)(cleared_lines * max_int(0, cleared_lines - 1));
        }

        if (!any_active_piece(env)) {
            draw_hand(env);
        }

        any_legal = update_observations(env);
        if (!any_legal) {
            reward += env->loss_penalty;
            env->terminals[0] = 1.0f;
            env->episode_return += reward;
            write_episode_log(env);
            start_episode(env);
            env->rewards[0] = reward;
            return;
        }
    }

    env->rewards[0] = reward;
    env->episode_return += reward;
}

static void draw_piece_preview(BlockPuzzle* env, int slot, int origin_x, int origin_y, int preview_cell) {
    PieceDef* piece = &BLOCK_PIECES[env->hand_piece[slot]];
    Color color = env->hand_active[slot] ? PUFF_FILL : PUFF_USED;
    int row;
    int col;

    for (row = 0; row < PREVIEW_SIZE; row++) {
        for (col = 0; col < PREVIEW_SIZE; col++) {
            Rectangle rect = {
                (float)(origin_x + col * preview_cell),
                (float)(origin_y + row * preview_cell),
                (float)(preview_cell - 2),
                (float)(preview_cell - 2),
            };
            DrawRectangleLinesEx(rect, 1.0f, PUFF_GRID);
            if (piece->preview_mask & BIT_AT(row, col)) {
                DrawRectangleRec(rect, color);
            }
        }
    }
}

static void c_render(BlockPuzzle* env) {
    const int cell_px = 36;
    const int preview_cell = 18;
    const int padding = 20;
    const int board_px = BOARD_SIZE * cell_px;
    const int width = padding * 2 + board_px;
    const int height = padding * 3 + board_px + 120;
    int row;
    int col;
    int slot;

    if (!BLOCK_PUZZLE_WINDOW_READY) {
        InitWindow(width, height, "PufferLib Block Puzzle");
        SetTargetFPS(30);
        BLOCK_PUZZLE_WINDOW_READY = true;
    }

    BeginDrawing();
    ClearBackground(PUFF_BG);

    for (row = 0; row < BOARD_SIZE; row++) {
        for (col = 0; col < BOARD_SIZE; col++) {
            Rectangle rect = {
                (float)(padding + col * cell_px),
                (float)(padding + row * cell_px),
                (float)(cell_px - 2),
                (float)(cell_px - 2),
            };
            DrawRectangleRec(rect, env->board[row][col] ? PUFF_FILL : PUFF_GHOST);
            DrawRectangleLinesEx(rect, 1.0f, PUFF_GRID);
        }
    }

    DrawText(TextFormat("Score: %d", env->score), padding, padding + board_px + 10, 20, PUFF_TEXT);
    DrawText(TextFormat("Lines: %d", env->lines_cleared), padding + 160, padding + board_px + 10, 20, PUFF_TEXT);
    DrawText(TextFormat("Pieces: %d", env->pieces_placed), padding + 320, padding + board_px + 10, 20, PUFF_TEXT);

    for (slot = 0; slot < HAND_SIZE; slot++) {
        const int px = padding + slot * (PREVIEW_SIZE * preview_cell + 24);
        const int py = padding + board_px + 40;
        DrawText(TextFormat("Piece %d", slot + 1), px, py - 18, 16, PUFF_TEXT);
        draw_piece_preview(env, slot, px, py, preview_cell);
    }

    EndDrawing();
}

static void c_close(BlockPuzzle* env) {
    (void)env;
    if (BLOCK_PUZZLE_WINDOW_READY && IsWindowReady()) {
        CloseWindow();
        BLOCK_PUZZLE_WINDOW_READY = false;
    }
}

#endif
