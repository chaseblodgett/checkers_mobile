# Piece values
MG_PIECE_VALUES = {
    'P': 82,
    'N': 337,
    'B': 365,
    'R': 477,
    'Q': 1025,
    'K': 0,
}

EG_PIECE_VALUES = {
    'P': 94,
    'N': 281,
    'B': 297,
    'R': 512,
    'Q': 936,
    'K': 0,
}

GAME_PHASE_INC = {
    'P': 0,  'N': 1, 'B': 1,
    'R': 2,  'Q': 4, 'K': 0
}

MG_PAWN_TABLE_BLACK = [
    [0,   0,   0,   0,   0,   0,  0,   0],
    [-11, 34, 126,  68,  95,  61, 134,  98],
    [-20, 25,  56,  65,  31,  26,   7,  -6],
    [-23, 17,  12,  23,  21,   6,  13, -14],
    [-25, 10,   6,  17,  12,  -5,  -2, -27],
    [-12, 33,   3,   3, -10,  -4,  -4, -26],
    [-22, 38,  24, -15, -23, -20,  -1, -35],
    [0,   0,   0,   0,   0,   0,  0,   0],
]

EG_PAWN_TABLE_BLACK = [
    [0,   0,   0,   0,   0,   0,  0,   0],
    [187, 165, 132, 147, 134, 158, 173, 178],
    [84,  82,  53,  56,  67,  85, 100,  94],
    [17,  17,   4,  -2,   5,  13,  24,  32],
    [-1,   3,  -8,  -7,  -7,  -3,   9,  13],
    [-8,  -1,  -5,   0,   1,  -6,   7,   4],
    [-7,   2,   0,  13,  10,  -1,  -3,  13],
    [0,   0,   0,   0,   0,   0,  0,   0],
]

MG_PAWN_TABLE_WHITE = [
    [0,   0,   0,   0,   0,   0,  0,   0],
    [-22, 38,  24, -15, -23, -20,  -1, -35],
    [-12, 33,   3,   3, -10,  -4,  -4, -26],
    [-25, 10,   6,  17,  12,  -5,  -2, -27],
    [-23, 17,  12,  23,  21,   6,  13, -14],
    [-20, 25,  56,  65,  31,  26,   7,  -6],
    [-11, 34, 126,  68,  95,  61, 134,  98],
    [0,   0,   0,   0,   0,   0,  0,   0],
]

EG_PAWN_TABLE_WHITE = [
    [0,   0,   0,   0,   0,   0,  0,   0],
    [-7,   2,   0,  13,  10,  -1,  -3,  13],
    [-8,  -1,  -5,   0,   1,  -6,   7,   4],
    [-1,   3,  -8,  -7,  -7,  -3,   9,  13],
    [17,  17,   4,  -2,   5,  13,  24,  32],
    [84,  82,  53,  56,  67,  85, 100,  94],
    [187, 165, 132, 147, 134, 158, 173, 178],
    [0,   0,   0,   0,   0,   0,  0,   0],
]

MG_KNIGHT_TABLE_BLACK = [
    [-107, -15, -97,  61, -49, -34, -89, -167],
    [-17,   7,  62,  23,  36,  72, -41,  -73],
    [44,  73, 129,  84,  65,  37,  60, -47],
    [22,  18,  69,  37,  53,  19,  17,  -9],
    [-8,  21,  19,  28,  13,  16,   4, -13],
    [-16,  25,  17,  19,  10,  12,  -9, -23],
    [-19, -14,  18,  -1,  -3, -12, -53, -29],
    [-23, -19, -28, -17, -33, -58, -21, -105],
]

EG_KNIGHT_TABLE_BLACK = [
    [-99, -63, -27, -31, -28, -13, -38, -58],
    [-52, -24, -25,  -9,  -2, -25,  -8, -25],
    [-41, -19,  -9,  -1,   9,  10, -20, -24],
    [-18,   8,  11,  22,  22,  22,   3, -17],
    [-18,   4,  17,  16,  25,  16,  -6, -18],
    [-22, -20,  -3,  10,  15,  -1,  -3, -23],
    [-44, -23, -20,  -2,  -5, -10, -20, -42],
    [-64, -50, -18, -22, -15, -23, -51, -29],
]

MG_KNIGHT_TABLE_WHITE = [
    [-23, -19, -28, -17, -33, -58, -21, -105],
    [-19, -14,  18,  -1,  -3, -12, -53, -29],
    [-16,  25,  17,  19,  10,  12,  -9, -23],
    [-8,  21,  19,  28,  13,  16,   4, -13],
    [22,  18,  69,  37,  53,  19,  17,  -9],
    [44,  73, 129,  84,  65,  37,  60, -47],
    [-17,   7,  62,  23,  36,  72, -41,  -73],
    [-107, -15, -97,  61, -49, -34, -89, -167],
]

EG_KNIGHT_TABLE_WHITE = [
    [-64, -50, -18, -22, -15, -23, -51, -29],
    [-44, -23, -20,  -2,  -5, -10, -20, -42],
    [-22, -20,  -3,  10,  15,  -1,  -3, -23],
    [-18,   4,  17,  16,  25,  16,  -6, -18],
    [-18,   8,  11,  22,  22,  22,   3, -17],
    [-41, -19,  -9,  -1,   9,  10, -20, -24],
    [-52, -24, -25,  -9,  -2, -25,  -8, -25],
    [-99, -63, -27, -31, -28, -13, -38, -58],
]


MG_BISHOP_TABLE_BLACK = [
    [-8, 7, -42, -25, -37, -82, 4, -29],
    [-47, 18, 59, 30, -13, -18, 16, -26],
    [-2, 37, 50, 35, 40, 43, 37, -16],
    [-2, 7, 37, 37, 50, 19, 5, -4],
    [4, 10, 12, 34, 26, 13, 13, -6],
    [10, 18, 27, 14, 15, 15, 15, 0],
    [1, 33, 21, 7, 0, 16, 15, 4],
    [-21, -39, -12, -13, -21, -14, -3, -33],
]

EG_BISHOP_TABLE_BLACK  = [
    [-24, -17, -9, -7, -8, -11, -21, -14],
    [-14, -4, -13, -3, -12, 7, -4, -8],
    [4, 0, 6, -2, -1, 0, -8, 2],
    [2, 3, 10, 14, 9, 12, 9, -3],
    [-9, -3, 10, 7, 19, 13, 3, -6],
    [-15, -7, 3, 13, 10, 8, -3, -12],
    [-27, -15, -9, 4, -1, -7, -18, -14],
    [-17, -5, -16, -9, -5, -23, -9, -23],
]

MG_BISHOP_TABLE_WHITE  = [
    [-21, -39, -12, -13, -21, -14, -3, -33],
    [1, 33, 21, 7, 0, 16, 15, 4],
    [10, 18, 27, 14, 15, 15, 15, 0],
    [4, 10, 12, 34, 26, 13, 13, -6],
    [-2, 7, 37, 37, 50, 19, 5, -4],
    [-16, 37, 50, 35, 40, 43, 37, -16],
    [-47, 18, 59, 30, -13, -18, 16, -26],
    [-8, 7, -42, -25, -37, -82, 4, -29],
]

EG_BISHOP_TABLE_WHITE = [
    [-17, -5, -16, -9, -5, -23, -9, -23],
    [-27, -15, -9, 4, -1, -7, -18, -14],
    [-15, -7, 3, 13, 10, 8, -3, -12],
    [-9, -3, 10, 7, 19, 13, 3, -6],
    [2, 3, 10, 14, 9, 12, 9, -3],
    [4, 0, 6, -2, -1, 0, -8, 2],
    [-14, -4, -13, -3, -12, 7, -4, -8],
    [-24, -17, -9, -7, -8, -11, -21, -14],
]


MG_ROOK_TABLE_BLACK= [
    [43, 31, 9, 63, 51, 32, 42, 32],
    [44, 26, 67, 80, 62, 58, 32, 27],
    [16, 61, 45, 17, 36, 26, 19, -5],
    [-20, -8, 35, 24, 26, 7, -11, -24],
    [-23, 6, -7, 9, -1, -12, -26, -36],
    [-33, -5, 0, 3, -17, -16, -25, -45],
    [-71, -6, 11, -1, -9, -20, -16, -44],
    [-26, -37, 7, 16, 17, 1, -13, -19],
]

EG_ROOK_TABLE_BLACK= [
    [5, 8, 12, 12, 15, 18, 10, 13],
    [3, 8, 3, -3, 11, 13, 13, 11],
    [-3, -5, -3, 4, 5, 7, 7, 7],
    [2, -1, 1, 2, 1, 13, 3, 4],
    [-11, -8, -6, -5, 4, 8, 5, 3],
    [-16, -8, -12, -7, -1, -5, 0, -4],
    [-3, -11, -9, -9, 2, 0, -6, -6],
    [-20, 4, -13, -5, -1, 3, 2, -9],
]

MG_ROOK_TABLE_WHITE = [
    [-26, -37, 7, 16, 17, 1, -13, -19],
    [-71, -6, 11, -1, -9, -20, -16, -44],
    [-33, -5, 0, 3, -17, -16, -25, -45],
    [-23, 6, -7, 9, -1, -12, -26, -36],
    [-20, -8, 35, 24, 26, 7, -11, -24],
    [16, 61, 45, 17, 36, 26, 19, -5],
    [44, 26, 67, 80, 62, 58, 32, 27],
    [43, 31, 9, 63, 51, 32, 42, 32],
]

EG_ROOK_TABLE_WHITE = [
    [-20, 4, -13, -5, -1, 3, 2, -9],
    [-3, -11, -9, -9, 2, 0, -6, -6],
    [-16, -8, -12, -7, -1, -5, 0, -4],
    [-11, -8, -6, -5, 4, 8, 5, 3],
    [2, -1, 1, 2, 1, 13, 3, 4],
    [-3, -5, -3, 4, 5, 7, 7, 7],
    [3, 8, 3, -3, 11, 13, 13, 11],
    [5, 8, 12, 12, 15, 18, 10, 13],
]


MG_QUEEN_TABLE_BLACK = [
    [45, 43, 44, 59, 12, 29, 0, -28],
    [54, 28, 57, -16, 1, -5, -39, -24],
    [57, 47, 56, 29, 8, 7, -17, -13],
    [1, -2, 17, -1, -16, -16, -27, -27],
    [-3, 3, -4, -2, -10, -9, -26, -9],
    [5, 14, 2, -5, -2, -11, 2, -14],
    [1, -3, 15, 8, 2, 11, -8, -35],
    [-50, -31, -25, -15, 10, -9, -18, -1],
]

EG_QUEEN_TABLE_BLACK = [
    [20, 10, 19, 27, 27, 22, 22, -9],
    [0, 30, 25, 58, 41, 32, 20, -17],
    [9, 19, 35, 47, 49, 9, 6, -20],
    [36, 57, 40, 57, 45, 24, 22, 3],
    [23, 39, 34, 31, 47, 19, 28, -18],
    [5, 10, 17, 9, 6, 15, -27, -16],
    [-32, -36, -23, -16, -16, -30, -23, -22],
    [-41, -20, -32, -5, -43, -22, -28, -33],
]

MG_QUEEN_TABLE_WHITE= [
    [-50, -31, -25, -15, 10, -9, -18, -1],
    [1, -3, 15, 8, 2, 11, -8, -35],
    [5, 14, 2, -5, -2, -11, 2, -14],
    [-3, 3, -4, -2, -10, -9, -26, -9],
    [1, -2, 17, -1, -16, -16, -27, -27],
    [57, 47, 56, 29, 8, 7, -17, -13],
    [54, 28, 57, -16, 1, -5, -39, -24],
    [45, 43, 44, 59, 12, 29, 0, -28],
]

EG_QUEEN_TABLE_WHITE = [
    [-41, -20, -32, -5, -43, -22, -28, -33],
    [-32, -36, -23, -16, -16, -30, -23, -22],
    [5, 10, 17, 9, 6, 15, -27, -16],
    [23, 39, 34, 31, 47, 19, 28, -18],
    [36, 57, 40, 57, 45, 24, 22, 3],
    [9, 19, 35, 47, 49, 9, 6, -20],
    [0, 30, 25, 58, 41, 32, 20, -17],
    [20, 10, 19, 27, 27, 22, 22, -9],
]

MG_KING_TABLE_BLACK = [
    [13,  2, -34, -56, -15, 16, 23, -65],
    [-29, -38, -4, -8, -7, -20, -1, 29],
    [-22, 22, 6, -20, -16, 2, 24, -9],
    [-36, -14, -25, -30, -27, -12, -20, -17],
    [-51, -33, -44, -46, -39, -27, -1, -49],
    [-27, -15, -30, -44, -46, -22, -14, -14],
    [8, 9, -16, -43, -64, -8, 7, 1],
    [14, 24, -28, 8, -54, 12, 36, -15],
]

EG_KING_TABLE_BLACK = [
    [-17, 4, 15, -11, -18, -18, -35, -74],
    [11, 23, 38, 17, 17, 14, 17, -12],
    [13, 44, 45, 20, 15, 23, 17, 10],
    [3, 26, 33, 26, 27, 24, 22, -8],
    [-11, 9, 23, 27, 24, 21, -4, -18],
    [-9, 7, 16, 23, 21, 11, -3, -19],
    [-17, -5, 4, 14, 13, 4, -11, -27],
    [-43, -24, -14, -28, -11, -21, -34, -53],
]


MG_KING_TABLE_WHITE = [
    [14, 24, -28, 8, -54, 12, 36, -15],
    [8, 9, -16, -43, -64, -8, 7, 1],
    [-27, -15, -30, -44, -46, -22, -14, -14],
    [-51, -33, -44, -46, -39, -27, -1, -49],
    [-36, -14, -25, -30, -27, -12, -20, -17],
    [-22, 22, 6, -20, -16, 2, 24, -9],
    [-29, -38, -4, -8, -7, -20, -1, 29],
    [13,  2, -34, -56, -15, 16, 23, -65],
]

EG_KING_TABLE_WHITE= [
    [-43, -24, -14, -28, -11, -21, -34, -53],
    [-17, -5, 4, 14, 13, 4, -11, -27],
    [-9, 7, 16, 23, 21, 11, -3, -19],
    [-11, 9, 23, 27, 24, 21, -4, -18],
    [3, 26, 33, 26, 27, 24, 22, -8],
    [13, 44, 45, 20, 15, 23, 17, 10],
    [11, 23, 38, 17, 17, 14, 17, -12],
    [-17, 4, 15, -11, -18, -18, -35, -74],
]


# Add similar tables for BISHOP, ROOK, QUEEN, KING

MG_PIECE_SQUARE_TABLES = {
    'WP': MG_PAWN_TABLE_WHITE,
    'BP': MG_PAWN_TABLE_BLACK,
    'WN': MG_KNIGHT_TABLE_WHITE,
    'BN': MG_KNIGHT_TABLE_BLACK,
    'WB': MG_BISHOP_TABLE_WHITE,
    'BB': MG_BISHOP_TABLE_BLACK,
    'WR': MG_ROOK_TABLE_WHITE,
    'BR': MG_ROOK_TABLE_BLACK,
    'WQ': MG_QUEEN_TABLE_WHITE,
    'BQ': MG_QUEEN_TABLE_BLACK,
    'WK': MG_KING_TABLE_WHITE,
    'BK': MG_KING_TABLE_BLACK,
}

EG_PIECE_SQUARE_TABLES = {
    'WP': EG_PAWN_TABLE_WHITE,
    'BP': EG_PAWN_TABLE_BLACK,
    'WN': EG_KNIGHT_TABLE_WHITE,
    'BN': EG_KNIGHT_TABLE_BLACK,
    'WB': EG_BISHOP_TABLE_WHITE,
    'BB': EG_BISHOP_TABLE_BLACK,
    'WR': EG_ROOK_TABLE_WHITE,
    'BR': EG_ROOK_TABLE_BLACK,
    'WQ': EG_QUEEN_TABLE_WHITE,
    'BQ': EG_QUEEN_TABLE_BLACK,
    'WK': EG_KING_TABLE_WHITE,
    'BK': EG_KING_TABLE_BLACK,
}


def evaluate_board(state):
    board = state.board
    mg = {'W': 0, 'B': 0}
    eg = {'W': 0, 'B': 0}
    game_phase = 0

    for (row, col), piece in board.items():
        if not piece:
            continue

        color = piece[0]
        piece_type = piece[1]

        # Piece values
        mg_value = MG_PIECE_VALUES[piece_type]
        eg_value = EG_PIECE_VALUES[piece_type]

        mg[color] += mg_value + MG_PIECE_SQUARE_TABLES[piece][row-1][col-1]
        eg[color] += eg_value + EG_PIECE_SQUARE_TABLES[piece][row-1][col-1]

        # Game phase increment (typically based on non-pawn material)
        if piece_type in ['Q', 'R', 'B', 'N']:
            game_phase += GAME_PHASE_INC[piece_type]

    # Clamp game phase
    if game_phase > 24:
        game_phase = 24
    mg_phase = game_phase
    eg_phase = 24 - game_phase

    # Final score: positive for White, negative for Black
    mg_score = mg['W'] - mg['B']
    eg_score = eg['W'] - eg['B']
    final_score = (mg_score * mg_phase + eg_score * eg_phase) / 24

    return final_score

def evaluate_board_with_check(state):
    board = state.board
    mg = {'W': 0, 'B': 0}
    eg = {'W': 0, 'B': 0}
    game_phase = 0

    for (row, col), piece in board.items():
        if not piece:
            continue

        color = piece[0]
        piece_type = piece[1]

        mg_value = MG_PIECE_VALUES[piece_type]
        eg_value = EG_PIECE_VALUES[piece_type]

        mg[color] += mg_value + MG_PIECE_SQUARE_TABLES[piece][row-1][col-1]
        eg[color] += eg_value + EG_PIECE_SQUARE_TABLES[piece][row-1][col-1]

        if piece_type in ['Q', 'R', 'B', 'N']:
            game_phase += GAME_PHASE_INC[piece_type]


    if game_phase > 24:
        game_phase = 24
    mg_phase = game_phase
    eg_phase = 24 - game_phase

    mg_score = mg['W'] - mg['B']
    eg_score = eg['W'] - eg['B']
    final_score = (mg_score * mg_phase + eg_score * eg_phase) / 24

    if is_in_check(state, 'B'):
        final_score += 30 
    if is_in_check(state, 'W'):
        final_score -= 30 

    return final_score


def is_in_check(state, color):
    board = state.board
    opponent = 'B' if color == 'W' else 'W'
    king_pos = None

    def in_bounds(x, y):
        return 1 <= x <= 8 and 1 <= y <= 8

    # 1. Find king position
    for (x, y), piece in board.items():
        if piece == f"{color}K":
            king_pos = (x, y)
            break

    if not king_pos:
        return False

    kx, ky = king_pos

    directions = {
        'N': [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)],
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'R': [(-1, 0), (1, 0), (0, -1), (0, 1)],
        'Q': [(-1, -1), (-1, 1), (1, -1), (1, 1),
              (-1, 0), (1, 0), (0, -1), (0, 1)],
        'K': [(-1, -1), (-1, 1), (1, -1), (1, 1),
              (-1, 0), (1, 0), (0, -1), (0, 1)],
    }

    for dx, dy in directions['N']:
        nx, ny = kx + dx, ky + dy
        if in_bounds(nx, ny):
            piece = board.get((nx, ny))
            if piece == f"{opponent}N":
                return True

    for piece_type in ['R', 'B', 'Q']:
        for dx, dy in directions[piece_type]:
            nx, ny = kx + dx, ky + dy
            while in_bounds(nx, ny):
                piece = board.get((nx, ny))
                if piece:
                    if piece[0] == opponent and piece[1] in (piece_type, 'Q'):
                        return True
                    break
                nx += dx
                ny += dy

    for dx, dy in directions['K']:
        nx, ny = kx + dx, ky + dy
        if in_bounds(nx, ny):
            piece = board.get((nx, ny))
            if piece == f"{opponent}K":
                return True

    pawn_dir = -1 if color == 'W' else 1
    for dy in [-1, 1]:
        nx, ny = kx + pawn_dir, ky + dy
        if in_bounds(nx, ny):
            piece = board.get((nx, ny))
            if piece == f"{opponent}P":
                return True

    return False
