from collections import namedtuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import time

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

class Checkers:
    """This class is based off the aima-python Game class. Refactored to be
    used with the game of Checkers. """

    def __init__(self, board=None, to_move=None):

        if board == None and to_move == None:
            self.board_size = 8
            board = {}
            for x in range(1, self.board_size + 1):
                for y in range(1, self.board_size + 1):

                    ## X's on the first 3 rows
                    if x <= 3 and (x + y) % 2 == 0:
                        board[(x, y)] = 'X'

                    ## O's on the back 3 rows
                    elif x >= 6 and (x + y) % 2 == 0:
                        board[(x, y)] = 'O'
                    else:
                        board[(x, y)] = None

            moves = []
            for (x, y), piece in board.items():
                if piece is not None and piece == "X":

                    possible_moves = self.get_non_jump_moves(board, (x, y), piece)
                    moves.append(((x, y), possible_moves))

            self.initial = GameState(to_move='X', utility=0, board=board, moves=moves)
        
        else:
            self.board_size = 8
            jumps = []

            for (x, y), piece in board.items():
                    if piece is not None and to_move in piece:

                        possible_jumps = self.get_piece_jumps(board, (x, y), piece)
                        jumps.append(((x,y), possible_jumps))

            if jumps:
                self.initial = GameState(to_move=to_move, utility=0, board=board,moves=jumps)

            else:

                moves = []
                for (x, y), piece in board.items():
                    if piece is not None and to_move in piece:

                        possible_moves = self.get_non_jump_moves(board, (x, y), piece)
                        possible_moves.append(((x,y), possible_jumps))

                self.initial = GameState(to_move=to_move, utility=0, board=board,moves=moves)


    def get_non_jump_moves(self, board, position, piece):
        """Calculate all possible non-jump moves for a piece at the given position."""
        x, y = position
        directions = []

        if piece == 'X':
            directions = [(1, -1), (1, 1)]  # Downward diagonals for X pieces
        elif piece == 'O':
            directions = [(-1, -1), (-1, 1)]  # Upward diagonals for O pieces
        elif piece in {'KX', 'KO'}:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All diagonals for King pieces

        non_jump_moves = []

        # Check criteria for normal moves
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 1 <= new_x <= self.board_size and 1 <= new_y <= self.board_size:
                if board.get((new_x, new_y)) is None:
                    non_jump_moves.append((new_x, new_y))

        return non_jump_moves

    def get_piece_jumps(self, board, position, piece):
        """Calculate all possible jumps for a specific piece at the given position."""

        x, y = position
        jumps = []
        directions = []

        if piece == 'X':
            directions = [(1, -1), (1, 1)]  # Downward diagonals for X pieces
        elif piece == 'O':
            directions = [(-1, -1), (-1, 1)]  # Upward diagonals for O pieces
        elif piece in {'KX', 'KO'}:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All diagonals for King pieces

        # Check for possible jumps
        for dx, dy in directions:
            mid_x, mid_y = x + dx, y + dy
            jump_x, jump_y = x + 2 * dx, y + 2 * dy

            # Ensure jump is within bounds
            if 1 <= jump_x <= self.board_size and 1 <= jump_y <= self.board_size:
                mid_piece = board.get((mid_x, mid_y))
                landing_square = board.get((jump_x, jump_y))

                # Check if the jump is valid
                if mid_piece and is_opposite_color(piece, mid_piece) and landing_square is None:
                    jumps.append(((jump_x, jump_y)))

        return jumps


    def actions(self, state):
        """Return a list of tuples of moves in form (curr_pos, target_pos)."""
        flat_moves = []
        for piece_position, possible_moves in state.moves:
            for move in possible_moves:
                flat_moves.append((piece_position, move))

        return flat_moves


    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        board = state.board.copy()
        (x1, y1), (x2, y2) = move
        print("Turn: ", state.to_move)

        ## Move piece to new square and remove it from old square
        piece = board[(x1, y1)]
        board[(x1, y1)] = None
        board[(x2, y2)] = piece

        # If the move is a jump
        if abs(x2 - x1) == 2 and abs(y2 - y1) == 2:
            # Remove jumped piece
            x_mid, y_mid = (x1 + x2) // 2, (y1 + y2) // 2
            board[(x_mid, y_mid)] = None

            # Check if king promotion
            if piece == 'X' and x2 == 8:
                board[(x2, y2)] = 'KX'
            elif piece == 'O' and x2 == 1:
                board[(x2, y2)] = 'KO'

            # Check for further jumps (i.e. double/triple jumps)
            further_jumps = []
            possible_jumps = self.get_piece_jumps(board, (x2, y2), piece)
            if possible_jumps:
                further_jumps.append(((x2, y2), possible_jumps))
                return GameState(to_move=state.to_move, utility=state.utility, board=board, moves=further_jumps)

        # Check if king promotion
        if piece == 'X' and x2 == 8:
            board[(x2, y2)] = 'KX'
        elif piece == 'O' and x2 == 1:
            board[(x2, y2)] = 'KO'

        # Switch to the next player if no further jumps are available
        next_player = 'O' if state.to_move == 'X' else 'X'

        # Check all pieces if jumps are possible
        jumps = []
        for (x, y), piece in board.items():
            if piece and (piece == next_player or piece == f'K{next_player}'):
                possible_moves = self.get_piece_jumps(board, (x, y), piece)
                if possible_moves:
                    jumps.append(((x, y), possible_moves))

        # Jumps are available so player must choose one of them
        if jumps:
            return GameState(to_move=next_player, utility=state.utility, board=board, moves=jumps)

        # Player has no jumps must make normal move
        moves = []
        for (x, y), piece in board.items():
            if piece and (piece == next_player or piece == f'K{next_player}'):
                possible_moves = self.get_non_jump_moves(board, (x, y), piece)
                if possible_moves:
                    moves.append(((x, y), possible_moves))

        return GameState(to_move=next_player, utility=state.utility, board=board, moves=moves)



    def utility(self, state, player):
        """Returns 1 for player ⏺ winning and -1 for player ◯ winning."""
        x_pieces = any(piece in {"X", "KX"} for piece in state.board.values())
        o_pieces = any(piece in {"O", "KO"} for piece in state.board.values())

        if not x_pieces:
            return -1
        elif not o_pieces:
            return 1
        else:
            return 1 if player == 'O' else -1

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        # Check if the current player has any legal moves

        # Check if any player has no pieces left
        x_pieces = any(piece in {"X", "KX"} for piece in state.board.values())
        o_pieces = any(piece in {"O", "KO"} for piece in state.board.values())

         # Game over if no pieces left
        if not x_pieces or not o_pieces:
            return True

        current_player = self.to_move(state)

        for (x, y), piece in state.board.items():
            if piece and (piece == current_player or piece == "K" + current_player):
                possible_jumps = self.get_piece_jumps(state.board, (x, y), piece)
                possible_moves = self.get_non_jump_moves(state.board, (x, y), piece)
                if possible_jumps or possible_moves:
                    return False

        # Game over if no moves are left
        return True



    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Display the board to the terminal."""
        board = state.board
        board_size = self.board_size

        border = "  +" + "---+" * board_size
        header = "    " + "   ".join(str(x) for x in range(1, board_size + 1))

        print(header)
        print(border)

        for row in range(1, board_size + 1):
            row_display = []
            for col in range(1, board_size + 1):
                piece = board.get((row, col))
                if (row + col) % 2 == 0:
                    if piece == "X":
                        row_display.append("⏺")
                    elif piece == "KX":
                        row_display.append("♕")
                    elif piece == "O":
                        row_display.append("◯")
                    elif piece == "KO":
                        row_display.append("♔")
                    else:
                        row_display.append(" ")
                else:
                    row_display.append(" ")

            print(f"{row:2} | " + " | ".join(row_display) + " |")
            print(border)

 
    def make_turn(self, state, move, ai_player_fn):
        """Apply player's move, then let AI move if game not over."""
        # Apply human move
        if move:
            state = self.result(state, move)

        # Check for game end after human move
        if self.terminal_test(state):
            return state, None 

        ai_move = None
        # AI move
        while state.to_move == 'X':
            ai_move = ai_player_fn(self, state)
            state = self.result(state, ai_move)

        return state, ai_move


    def play_game(self, *players):
        """Play a 2-person, move-alternating game."""
        state = self.initial
        current_player = self.to_move(state)

        while True:
            for player in players:
                current_player = self.to_move(state)
                while current_player == self.to_move(state):

                    move = player(self, state)
                    state = self.result(state, move)
                    if self.terminal_test(state):
                        self.display(state)
                        return self.utility(state, self.to_move(state))

                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(state))

    def play_game_with_timeout(self, *players):
        """Play a 2-person, move-alternating game."""
        start_time = time.time()
        state = self.initial
        current_player = self.to_move(state)

        while True:
            elapsed_time = time.time() - start_time
            if(elapsed_time > 30):
                self.display(state)
                return 0
            for player in players:

                current_player = self.to_move(state)
                while current_player == self.to_move(state):

                    move = player(self, state)
                    state = self.result(state, move)
                    if self.terminal_test(state):
                        self.display(state)
                        return self.utility(state, self.to_move(state))

                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(state))


def is_opposite_color(piece1, piece2):
    """Used to check whether a piece is able to jump over another piece"""
    if piece1 is None or piece2 is None:
        return False
    color_map = {
        'X': ['O', 'KO'],
        'KX': ['O', 'KO'],
        'O': ['X', 'KX'],
        'KO': ['X', 'KX']
    }
    return piece2 in color_map.get(piece1, [])

### THIS FUNCTION IS TAKEN FROM THE AIMA PYTHON CODE, WE DO NOT OWN THIS FUNCTION
def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):

    """THE CODE FOR THIS FUNCTION IS TAKEN FROM THE AIMA PYTHON CODE, WE DO NOT OWN THIS FUNCTION
    """
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def query_player(game, state):
    """Human player. Can play as X or O, will prompt user for a move"""
    print("Current state:")
    game.display(state)

    current_player = state.to_move
    if current_player == "X":
        symbol = "⏺"
    else:
        symbol = "◯"
    move_options = []
    print(f"\nAvailable moves for player {symbol}:")
    move_index = 1

    # Get all possible moves for the current player
    for move in game.actions(state):
        piece_position, target_position = move
        piece = state.board[piece_position]
        if piece == current_player or piece == f'K{current_player}':
            print(f"{move_index}: {piece_position} --> {target_position}")
            move_options.append(move)
            move_index += 1

    if not move_options:
        print('No legal moves: passing turn to the next player.')
        return None


    # Get move from user
    move = None
    while move is None:
        try:
            move_number = int(input("\nEnter the number of the move you want to play: ")) - 1
            if 0 <= move_number < len(move_options):
                move = move_options[move_number]
            else:
                print("Invalid move number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid move number.")

    return move


# Player X eval for counting pieces where kings are worth the same
def count_pieces_eval_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            if piece == "X" or piece == "KX":
                eval += 0.01
            elif piece == "O" or piece == "KO":
                eval -= 0.02
    return eval

# Player O eval for counting pieces where kings are worth the same
def count_pieces_eval_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            if piece == "X" or piece == "KX":
                eval -= 0.02
            elif piece == "O" or piece == "KO":
                eval += 0.01
    return eval

# Player O eval for counting pieces where kings are weighted more
def count_pieces_eval_with_king_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            if piece == "X":
                eval -= 0.04
            elif piece == "KX":
                eval -= 0.07
            elif piece == "O":
                eval += 0.01
            elif piece == "KO":
                eval += 0.02
    return eval

# Player X eval for counting pieces where kings are weighted more
def count_pieces_eval_with_king_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            if piece == "X":
                eval += 0.01
            elif piece == "KX":
                eval += 0.02
            elif piece == "O":
                eval -= 0.04
            elif piece == "KO":
                eval -= 0.07
    return eval

"""Beginning of added/renamed evals"""
# Player O eval with center control and clustering
def eval_with_center_control_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval -= 0.02
            elif piece == "O" or piece == "KO":
                eval += 0.01

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

    return eval

def eval_with_center_control_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval += 0.01
            elif piece == "O" or piece == "KO":
                eval -= 0.02

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02

    return eval

def eval_with_king_center_control_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval -= 0.05
            elif piece == "KX":
                eval -= 0.10
            elif piece == "O":
                eval += 0.02
            elif piece == "KO":
                eval += 0.05

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

    return eval

def eval_with_king_center_control_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval += 0.02
            elif piece == "KX":
                eval += 0.05
            elif piece == "O":
                eval -= 0.05
            elif piece == "KO":
                eval -= 0.10

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02
    return eval

def eval_cluster_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval -= 0.02
            elif piece == "O" or piece == "KO":
                eval += 0.01

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

            # Penalize for clustering pieces in adjacent squares
            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "O" and adjacent_piece == "O":
                    eval -= 0.01
                elif piece == "KO" and adjacent_piece == "KO":
                    eval -= 0.02
    return eval

def eval_cluster_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval += 0.01
            elif piece == "O" or piece == "KO":
                eval -= 0.02

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02

            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            # Penalize for clustering pieces in adjacent squares
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "X" and adjacent_piece == "X":
                    eval -= 0.01
                elif piece == "KX" and adjacent_piece == "KX":
                    eval -= 0.02
    return eval

def eval_cluster_king_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval -= 0.05
            elif piece == "KX":
                eval -= 0.10
            elif piece == "O":
                eval += 0.02
            elif piece == "KO":
                eval += 0.05

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

            # Penalize for clustering pieces in adjacent squares
            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "O" and adjacent_piece == "O":
                    eval -= 0.01
                elif piece == "KO" and adjacent_piece == "KO":
                    eval -= 0.02
    return eval

def eval_cluster_king_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval += 0.02
            elif piece == "KX":
                eval += 0.05
            elif piece == "O":
                eval -= 0.05
            elif piece == "KO":
                eval -= 0.10

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02

            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            # Penalize for clustering pieces in adjacent squares
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "X" and adjacent_piece == "X":
                    eval -= 0.01
                elif piece == "KX" and adjacent_piece == "KX":
                    eval -= 0.02
    return eval

def eval_almost_complete_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval -= 0.02
            elif piece == "O" or piece == "KO":
                eval += 0.01

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

             # Slight reward for advancing pieces down the board
            if piece == "O":
                eval += 0.005 * row
            elif piece == "KO":
                eval += 0.01

            # Penalize for clustering pieces in adjacent squares
            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "O" and adjacent_piece == "O":
                    eval -= 0.01
                elif piece == "KO" and adjacent_piece == "KO":
                    eval -= 0.02
    return eval

def eval_almost_complete_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X" or piece == "KX":
                eval += 0.01
            elif piece == "O" or piece == "KO":
                eval -= 0.02

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02

            # Slight reward for advancing pieces down the board
            if piece == "X":
                eval += 0.005 * (9 - row)
            elif piece == "KX":
                eval += 0.01


            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            # Penalize for clustering pieces in adjacent squares
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "X" and adjacent_piece == "X":
                    eval -= 0.01
                elif piece == "KX" and adjacent_piece == "KX":
                    eval -= 0.02
    return eval

# Player O eval with king weighted and center control
def eval_complete_O(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval -= 0.05
            elif piece == "KX":
                eval -= 0.10
            elif piece == "O":
                eval += 0.02
            elif piece == "KO":
                eval += 0.05

             # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval -= 0.01
                elif piece == "KX":
                    eval -= 0.02
                elif piece == "O":
                    eval += 0.02
                elif piece == "KO":
                    eval += 0.03

             # Slight reward for advancing pieces down the board
            if piece == "O":
                eval += 0.005 * row
            elif piece == "KO":
                eval += 0.01

            # Penalize for clustering pieces in adjacent squares
            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "O" and adjacent_piece == "O":
                    eval -= 0.01
                elif piece == "KO" and adjacent_piece == "KO":
                    eval -= 0.02
    return eval

# Player X eval with king weighted and center control
def eval_complete_X(state):
    eval = 0
    for row in range(1, 9):
        for col in range(1, 9):
            piece = state.board.get((row, col))
            # Base piece evaluation
            if piece == "X":
                eval += 0.02
            elif piece == "KX":
                eval += 0.05
            elif piece == "O":
                eval -= 0.05
            elif piece == "KO":
                eval -= 0.10

            # Reward for central board control
            if 3 <= row <= 6 and 3 <= col <= 6:
                if piece == "X":
                    eval += 0.02
                elif piece == "KX":
                    eval += 0.03
                elif piece == "O":
                    eval -= 0.01
                elif piece == "KO":
                    eval -= 0.02

            # Slight reward for advancing pieces down the board
            if piece == "X":
                eval += 0.005 * (9 - row)
            elif piece == "KX":
                eval += 0.01


            adjacent_positions = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            # Penalize for clustering pieces in adjacent squares
            for adj_row, adj_col in adjacent_positions:
                adjacent_piece = state.board.get((adj_row, adj_col))
                if piece == "X" and adjacent_piece == "X":
                    eval -= 0.01
                elif piece == "KX" and adjacent_piece == "KX":
                    eval -= 0.02
    return eval
"""End of added/renamed evals"""

## Different players X and O with different eval functions at different depths

# Player X with simple eval of counting pieces at different depths
def count_pieces_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=count_pieces_eval_X)

def count_pieces_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=count_pieces_eval_X)

def count_pieces_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=count_pieces_eval_X)

def count_pieces_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=count_pieces_eval_X)

# Player O with simple eval of counting pieces at different depths
def count_pieces_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=count_pieces_eval_O)

def count_pieces_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=count_pieces_eval_O)

def count_pieces_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=count_pieces_eval_O)

def count_pieces_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=count_pieces_eval_O)

# Player X with eval of counting pieces with kings weighted at different depths
def count_pieces_with_king_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=count_pieces_eval_with_king_X)

def count_pieces_with_king_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=count_pieces_eval_with_king_X)

def count_pieces_with_king_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=count_pieces_eval_with_king_X)

def count_pieces_with_king_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=count_pieces_eval_with_king_X)

# Player O with eval of counting pieces with kings weighted at different depths
def count_pieces_with_king_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=count_pieces_eval_with_king_O)

def count_pieces_with_king_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=count_pieces_eval_with_king_O)

def count_pieces_with_king_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=count_pieces_eval_with_king_O)

def count_pieces_with_king_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=count_pieces_eval_with_king_O)

def count_pieces_with_king_O_d5(game, state):
    return alpha_beta_cutoff_search(state, game, d=5, eval_fn=count_pieces_eval_with_king_O)

def count_pieces_with_king_O_d6(game, state):
    return alpha_beta_cutoff_search(state, game, d=6, eval_fn=count_pieces_eval_with_king_O)

"""Beginning of new players"""
# Player O with simple eval and center
def count_simple_center_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_with_center_control_O)

def count_simple_center_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_with_center_control_O)

def count_simple_center_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_with_center_control_O)

def count_simple_center_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_with_center_control_O)

# Player X with simple eval and center
def count_simple_center_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_with_center_control_X)

def count_simple_center_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_with_center_control_X)

def count_simple_center_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_with_center_control_X)

def count_simple_center_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_with_center_control_X)

# Player O with king eval and center
def count_king_center_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_with_king_center_control_O)

def count_king_center_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_with_king_center_control_O)

def count_king_center_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_with_king_center_control_O)

def count_king_center_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_with_king_center_control_O)

# Player X with King eval and center
def count_king_center_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_with_king_center_control_X)

def count_king_center_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_with_king_center_control_X)

def count_king_center_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_with_king_center_control_X)

def count_king_center_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_with_king_center_control_X)

# Player O with simple eval, center, cluster control
def count_simple_cluster_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_cluster_O)

def count_simple_cluster_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_cluster_O)

def count_simple_cluster_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_cluster_O)

def count_simple_cluster_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_cluster_O)

# Player X with simple eval, center, cluster control
def count_simple_cluster_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_cluster_X)

def count_simple_cluster_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_cluster_X)

def count_simple_cluster_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_cluster_X)

def count_simple_cluster_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_cluster_X)

# Player O with king eval, center, cluster control
def count_king_cluster_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_cluster_king_O)

def count_king_cluster_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_cluster_king_O)

def count_king_cluster_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_cluster_king_O)

def count_king_cluster_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_cluster_king_O)

# Player X with king eval, center, cluster control
def count_king_cluster_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_cluster_king_X)

def count_king_cluster_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_cluster_king_X)

def count_king_cluster_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_cluster_king_X)

def count_king_cluster_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_cluster_king_X)

# Player O complete functionality except king
def count_simple_complete_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_almost_complete_O)

def count_simple_complete_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_almost_complete_O)

def count_simple_complete_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_almost_complete_O)

def count_simple_complete_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_almost_complete_O)

# Player X complete functionality except king
def count_simple_complete_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_almost_complete_X)

def count_simple_complete_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_almost_complete_X)

def count_simple_complete_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_almost_complete_X)

def count_simple_complete_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_almost_complete_X)

# Player O complete functionality
def count_complete_O_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_complete_O)

def count_complete_O_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_complete_O)

def count_complete_O_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_complete_O)

def count_complete_O_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_complete_O)

def count_complete_O_d5(game, state):
    return alpha_beta_cutoff_search(state, game, d=5, eval_fn=eval_complete_O)

def count_complete_O_d6(game, state):
    return alpha_beta_cutoff_search(state, game, d=6, eval_fn=eval_complete_O)

# Player X complete functionality
def count_complete_X_d1(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=eval_complete_X)

def count_complete_X_d2(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=eval_complete_X)

def count_complete_X_d3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=eval_complete_X)

def count_complete_X_d4(game, state):
    return alpha_beta_cutoff_search(state, game, d=4, eval_fn=eval_complete_X)

def count_complete_X_d6(game, state):
    return alpha_beta_cutoff_search(state, game, d=6, eval_fn=eval_complete_X)
"""End of new players"""

# Random player
def random_player(game, state):
    """Returns a random move for the current player."""
    current_player = state.to_move

    valid_moves = [
        move for move in game.actions(state)
        if state.board[move[0]] in {current_player, f'K{current_player}'}
    ]

    return random.choice(valid_moves) if valid_moves else None


def tournament(x_players, o_players, game_class, num_rounds=1, timeout=30):
    """
    Conducts a tournament between X players and O players.
    """
    results = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_time": 0,
        "total_moves": 0,
        "total_moves_time": 0,
        "total_thinking_time": 0
    })
    all_match_results = []

    for round_num in range(num_rounds):
        print(f"--- Round {round_num + 1} ---")
        for name1, player1 in x_players:
            for name2, player2 in o_players:
                print(f"Match: {name1} (X) vs {name2} (O)")
                game = game_class()

                # Play new match with different players
                result, game_time, move_count, thinking_time_x, thinking_time_o = play_game_with_timeout(game, player1, player2, timeout)

                if result > 0:  # X player wins
                    print("Result: " + name1 + " wins!")
                    results[name1]["wins"] += 1
                    results[name2]["losses"] += 1
                elif result < 0:  # O player wins
                    print("Result: " + name2 + " wins!")
                    results[name1]["losses"] += 1
                    results[name2]["wins"] += 1
                else:  # Draw
                    results[name1]["draws"] += 1
                    results[name2]["draws"] += 1

                # Record metrics
                results[name1]["total_time"] += game_time
                results[name2]["total_time"] += game_time
                results[name1]["total_moves"] += move_count
                results[name2]["total_moves"] += move_count
                results[name1]["total_moves_time"] += game_time / max(1, move_count)
                results[name2]["total_moves_time"] += game_time / max(1, move_count)
                player_x_moves = move_count // 2 + (1 if move_count % 2 == 1 else 0) 
                player_o_moves = move_count // 2

                results[name1]["total_thinking_time"] += thinking_time_x / max(1, player_x_moves)
                results[name2]["total_thinking_time"] += thinking_time_o / max(1, player_o_moves)

                all_match_results.append((name1, name2, result, game_time, move_count, thinking_time_x, thinking_time_o))

    return results, all_match_results


def play_game_with_timeout(game, player_x, player_o, timeout):
    """
    Plays a game between two players with a timeout, tracking thinking time per player.

    Parameters:
    - game: The game instance.
    - player_x: Function for player X.
    - player_o: Function for player O.
    - timeout (int): Maximum allowed time for the game in seconds.

    Returns:
    - result (int): Utility value of the game state.
    - game_time (float): Total time taken for the game.
    - move_count (int): Total moves made in the game.
    """
    start_time = time.time()
    state = game.initial
    move_count = 0
    thinking_time_x = 0
    thinking_time_o = 0

    while not game.terminal_test(state):
        elapsed_time = time.time() - start_time
        # Timeout results in a draw
        if elapsed_time > timeout:
            game.display(state)
            print("Game timeout! Declaring a draw.")
            return 0, elapsed_time, move_count, thinking_time_x, thinking_time_o

        current_player = state.to_move
        move_start_time = time.time()

        if current_player == "X":
            action = player_x(game, state)
        else:
            action = player_o(game, state)

        move_end_time = time.time()

        # Update thinking time for each player
        if current_player == "X":
            thinking_time_x += move_end_time - move_start_time
        else:
            thinking_time_o += move_end_time - move_start_time

        state = game.result(state, action)
        move_count += 1

    # Determine the result
    result = game.utility(state, state.to_move)
    game.display(state)
    game_time = time.time() - start_time

    return result, game_time, move_count, thinking_time_x, thinking_time_o


def plot_tournament_results(results, num_rounds=1):
    """
    Plots the results of the tournament using matplotlib.
    """
    player_names = list(results.keys())
    wins = [results[player]["wins"] for player in player_names]
    losses = [results[player]["losses"] for player in player_names]
    draws = [results[player]["draws"] for player in player_names]
    avg_time = [results[player]["total_time"] / max(1, (wins[i] + losses[i] + draws[i])) for i, player in enumerate(player_names)]
    avg_moves = [results[player]["total_moves"] / max(1, (wins[i] + losses[i] + draws[i])) for i, player in enumerate(player_names)]
    avg_time_per_move = [results[player]["total_moves_time"] / max(1, results[player]["total_moves"]) for player in player_names]
    total_thinking_time = [results[player]["total_thinking_time"] for player in player_names]

    x = range(len(player_names))

    # Bar chart for wins, losses, and draws
    plt.figure(figsize=(12, 6))
    plt.bar(x, wins, width=0.2, label="Wins", color="g", align="center")
    plt.bar([p + 0.2 for p in x], losses, width=0.2, label="Losses", color="r", align="center")
    plt.bar([p - 0.2 for p in x], draws, width=0.2, label="Draws", color="b", align="center")
    plt.xticks(x, player_names, rotation=45, ha="right")
    plt.title(f"Tournament Results (Wins, Losses, Draws) - {num_rounds} Rounds")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Line chart for average moves per game
    plt.figure(figsize=(12, 6))
    plt.plot(player_names, avg_moves, label="Avg Moves", marker="s", color="orange", linestyle=":")
    plt.xticks(range(len(player_names)), player_names, rotation=45, ha="right")
    plt.title("Average Moves per Game")
    plt.ylabel("Average Moves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Line chart for average time per game
    plt.figure(figsize=(12, 6))
    plt.plot(player_names, avg_time, label="Avg Time (s)", marker="o", color="purple", linestyle="--")
    plt.xticks(range(len(player_names)), player_names, rotation=45, ha="right")
    plt.title("Average Time per Game")
    plt.ylabel("Average Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar chart for average thinking time per move
    plt.figure(figsize=(12, 6))
    plt.bar(x, total_thinking_time, label="Avg Time(s)", color="purple", align="center")
    plt.xticks(x, player_names, rotation=45, ha="right")
    plt.title("Average Thinking Time per Move")
    plt.ylabel("Average Thinking Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    x_players = [
        ("Random Player X", random_player),
        ("Count Complete X d2", count_complete_X_d1),
        ("Count Complete X d2", count_complete_X_d2),
        ("Count Pieces X d1", count_pieces_X_d1),
        ("Count Pieces X d2", count_pieces_X_d2),
        ("Count Pieces with King X d1", count_pieces_with_king_X_d1),
        ("Count Pieces with King X d2", count_pieces_with_king_X_d2),
        ("Count Simple Center X d1", count_simple_center_X_d1),
        ("Count Simple Center X d2", count_simple_center_X_d2),
        ("Count King Center X d1", count_king_center_X_d1),
        ("Count King Center X d2", count_king_center_X_d2),
        ("Count Simple Cluster X d1", count_simple_cluster_X_d1),
        ("Count Simple Cluster X d2", count_simple_cluster_X_d2),
        ("Count King Cluster X d1", count_king_cluster_X_d1),
        ("Count King Cluster X d2", count_king_cluster_X_d2),
        ("Count Simple Complete X d1", count_simple_complete_X_d1),
        ("Count Simple Complete X d2", count_simple_complete_X_d2),
        
    ]
    o_players = [
        ("Count Complete O d1", count_complete_O_d1),
        ("Count Complete O d2", count_complete_O_d2),
        # ("Count Complete O d3", count_complete_O_d3),
        # ("Count Complete O d4", count_complete_O_d4),
        # ("Count Complete O d5", count_complete_O_d5),
        # ("Count Complete O d6", count_complete_O_d6),
        ("Count Simple Complete O d1", count_simple_complete_O_d1),
        ("Count Simple Complete O d2", count_simple_complete_O_d2),
        ("Count King Cluster O d1", count_king_cluster_O_d1),
        ("Count King Cluster O d2", count_king_cluster_O_d2),
        ("Count Simple Cluster O d1", count_simple_cluster_O_d1),
        ("Count Simple Cluster O d2", count_simple_cluster_O_d2),
        ("Count King Center O d1", count_king_center_O_d1),
        ("Count King Center O d2", count_king_center_O_d2),
        ("Count Simple Center O d1", count_simple_center_O_d1),
        ("Count Simple Center O d2", count_simple_center_O_d2),
        ("Count Pieces with King O d1", count_pieces_with_king_O_d1),
        ("Count Pieces with King O d2", count_pieces_with_king_O_d2),
        # ("Count Pieces with King O d3", count_pieces_with_king_O_d3),
        # ("Count Pieces with King O d4", count_pieces_with_king_O_d4),
        # ("Count Pieces with King O d5", count_pieces_with_king_O_d5),
        # ("Count Pieces with King O d6", count_pieces_with_king_O_d6),
        ("Count Pieces O d1", count_pieces_O_d1),
        ("Count Pieces O d2", count_pieces_O_d2),
        ("Random Player O", random_player),
    ]
    game_class = Checkers

    print("Welcome! Please choose an option:")
    print("1. Play against an AI")
    print("2. Run a tournament")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # User chooses to play against an AI
        print("Choose AI difficulty:")
        print("1. Beginner")
        print("2. Intermediate")
        print("3. Advanced")
        print("4. Expert")

        difficulty = input("Enter the difficulty level (1, 2, 3, or 4): ").strip()
        if difficulty == "1":
            ai_player = random_player
            print("You selected Beginner AI.")
        elif difficulty == "2":
            ai_player = count_pieces_X_d2
            print("You selected Intermediate AI.")
        elif difficulty == "3":
            ai_player = count_pieces_with_king_X_d4
            print("You selected Advanced AI.")
        elif difficulty == "4":
            ai_player = count_complete_X_d6
            print("You selected Expert AI.")
        else:
            print("Invalid choice. Defaulting to Beginner AI.")
            ai_player = random_player

        # Start a game with the selected difficulty
        checkers = game_class()
        result = checkers.play_game(ai_player, query_player)

        if result > 0:
            print("Player ⏺ wins!")
        elif result < 0:
            print("Player ◯ wins!")
        else:
            print("Draw by timeout!")

    elif choice == "2":
        # User chooses to run a tournament
        print("Starting a tournament...")
        print("WARNING: Tournament may take a few minutes to complete.")
        time.sleep(3)
        num_rounds = 1
        timeout = 5

        results, match_results = tournament(x_players, o_players, game_class, num_rounds=num_rounds, timeout=timeout)

        plot_tournament_results(results, num_rounds=num_rounds)

        print("Tournament Complete!")
        for player, metrics in results.items():
            print(f"{player}: {metrics}")
    else:
        print("Invalid choice. Exiting the program.")


if __name__ == '__main__':
    main()