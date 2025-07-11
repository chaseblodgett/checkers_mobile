from collections import namedtuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from copy import deepcopy
import evaluation

GameState = namedtuple('GameState', 'to_move, utility, board, moves, castling_rights, en_passant_square, moves_last_capture')

class Chess:
    """This class is based off the aima-python Game class. Refactored to be
    used with the game of Chess. """

    def __init__(self, moves_last_capture=0, board=None, to_move=None, castling_rights=None, en_passant_square=None):

        if board == None and to_move == None:
            self.board_size = 8
            board = {}
            for x in range(1, self.board_size + 1):
                for y in range(1, self.board_size + 1):

                    if x == 2:
                        board[(x, y)] = 'WP'
                    elif x == 7:
                        board[(x, y)] = 'BP'

                    elif x == 1 and y == 4:
                        board[(x, y)] = 'WK'
                    elif x == 8 and y == 4:
                        board[(x, y)] = 'BK'

                    elif x == 1 and y == 5:
                        board[(x, y)] = 'WQ'
                    elif x == 8 and y == 5:
                        board[(x, y)] = 'BQ'

                    elif x == 1 and (y == 1 or y == 8):
                        board[(x, y)] = 'WR'
                    elif x == 8 and (y == 1 or y == 8):
                        board[(x, y)] = 'BR'
                    
                    elif x == 1 and (y == 3 or y == 6):
                        board[(x, y)] = 'WB'
                    elif x == 8 and (y == 3 or y == 6):
                        board[(x, y)] = 'BB'

                    elif x == 1 and (y == 2 or y == 7):
                        board[(x, y)] = 'WN'
                    elif x == 8 and (y == 2 or y == 7):
                        board[(x, y)] = 'BN'

                    else:
                        board[(x, y)] = None

            moves = []
            for (x, y), piece in board.items():
                if piece is not None and "W" in piece:

                    possible_moves = self.get_moves(board, (x, y), piece)
                    if possible_moves:
                        moves.append(((x, y), possible_moves))

            castling_rights = {'W': {'K': True, 'Q': True}, 'B': {'K': True, 'Q': True}}
            en_passant_square = None

            self.initial = GameState(to_move='W', utility=0, board=board, moves=moves, castling_rights=castling_rights, en_passant_square=en_passant_square, moves_last_capture=0)
        
        else:
            self.board_size = 8
            moves = []
            for (x, y), piece in board.items():

                if piece is not None and to_move == piece[0]:
                    
                    possible_moves = self.get_moves(board, (x, y), piece, castling_rights=castling_rights, en_passant_square=en_passant_square)
                    if possible_moves:
                        moves.append(((x, y), possible_moves))

            self.initial = GameState(to_move=to_move, utility=0, board=board,moves=moves, castling_rights=castling_rights, en_passant_square=en_passant_square, moves_last_capture=moves_last_capture)


    def get_moves(self, board, position, piece, castling_rights=None, en_passant_square=None, validate_checks=True):
        """Return a list of valid moves for a given chess piece at a position (1-indexed)."""
        x, y = position
        color = piece[0]
        piece_type = piece[1]
        moves = []

        def in_bounds(nx, ny):
            return 1 <= nx <= 8 and 1 <= ny <= 8

        def is_opponent(nx, ny):
            return board.get((nx, ny), '') and board[(nx, ny)][0] != color

        if piece_type == 'P':  # Pawn
            direction = 1 if color == 'W' else -1 
            start_row = 2 if color == 'W' else 7

            # Forward move
            if in_bounds(x + direction, y) and board.get((x + direction, y)) is None:
                moves.append((x + direction, y))
                # Double move from start
                if x == start_row and board.get((x + 2 * direction, y)) is None:
                    moves.append((x + 2 * direction, y))

            # Diagonal captures
            for dy in [-1, 1]:
                nx, ny = x + direction, y + dy
                if in_bounds(nx, ny):
                    if is_opponent(nx, ny):
                        moves.append((nx, ny))
                    elif (nx, ny) == en_passant_square:
                        moves.append((nx, ny))

        elif piece_type == 'R':  # Rook
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while in_bounds(nx, ny):
                    if board.get((nx, ny)) is None:
                        moves.append((nx, ny))
                    elif is_opponent(nx, ny):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy

        elif piece_type == 'B':  # Bishop
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while in_bounds(nx, ny):
                    if board.get((nx, ny)) is None:
                        moves.append((nx, ny))
                    elif is_opponent(nx, ny):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy

        elif piece_type == 'Q':  # Queen
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while in_bounds(nx, ny):
                    if board.get((nx, ny)) is None:
                        moves.append((nx, ny))
                    elif is_opponent(nx, ny):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy

        elif piece_type == 'K':
            # Normal king movement
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny):
                    if board.get((nx, ny)) is None or is_opponent(nx, ny):
                        moves.append((nx, ny))

            # Castling logic
            if castling_rights and color in castling_rights:
                rank = 1 if color == 'W' else 8
                kingside_clear = board.get((rank, 2)) is None and board.get((rank, 3)) is None
                queenside_clear = (board.get((rank, 5)) is None and
                                board.get((rank, 6)) is None and
                                board.get((rank, 7)) is None)

                if (castling_rights[color].get('K') and kingside_clear
                        and not self.is_check(board, color, (position, (rank, 2)))
                        and not self.is_check(board, color, (position, (rank, 3)))) :
                    moves.append((rank, 2))

                if (castling_rights[color].get('Q') and queenside_clear
                        and not self.is_check(board, color, (position, (rank, 5)))
                        and not self.is_check(board, color, (position, (rank, 6)))):
                    moves.append((rank, 6))  


        elif piece_type == 'N': 
            jumps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
            for dx, dy in jumps:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny):
                    if board.get((nx, ny)) is None or is_opponent(nx, ny):
                        moves.append((nx, ny))

        if validate_checks:
            legal_moves = []
            for end_pos in moves:
                if not self.is_check(board, color, (position, end_pos)):
                    legal_moves.append(end_pos)
            return legal_moves


        
        return moves


    def is_check(self, board, color, move):
        """Returns True if `color` is in check after applying `move` to the board."""

        opponent_color = 'B' if color == 'W' else 'W'
        new_board = deepcopy(board)

        start_pos, end_pos = move
        moving_piece = new_board.get(start_pos)
        new_board[end_pos] = moving_piece
        new_board[start_pos] = None

        king_position = None
        for pos, piece in new_board.items():
            if piece == f'{color}K':
                king_position = pos
                break

        if not king_position:
            return False  

        for pos, piece in new_board.items():
            if piece and piece[0] == opponent_color:
                possible_moves = self.get_moves(new_board, pos, piece, validate_checks=False)
                if king_position in possible_moves:
                    return True

        return False
    
    def is_stalemate(self, board, color):
        opponent_color = 'B' if color == 'W' else 'W'
        king_position = None
        for pos, piece in board.items():
            if piece == f'{color}K':
                king_position = pos
                break

        if not king_position:
            return False  

        for pos, piece in board.items():
            if piece and piece[0] == opponent_color:
                possible_moves = self.get_moves(board, pos, piece, validate_checks=False)
                if king_position in possible_moves:
                    return False

        return True


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
        new_castling_rights = deepcopy(state.castling_rights)
        new_en_passant_square = None
        moves_last_capture = state.moves_last_capture
        (x1, y1), (x2, y2) = move
        piece = board[(x1, y1)]

        if piece[1] =='P' or board[(x2, y2)] is not None:
            moves_last_capture = 0

        board[(x1, y1)] = None
        board[(x2, y2)] = piece

        # Check if pawn promotion (promote to queen by default)
        if piece[1] == 'P' and x2 == 8:
            board[(x2, y2)] = 'WQ'
        elif piece[1] == 'P' and x2 == 1:
            board[(x2, y2)] = 'BQ'
        
        if piece[1] == 'P' and (x2, y2) == state.en_passant_square:
            if piece[0] == 'W':
                board[(x2 + 1, y2)] = None
            else:
                board[(x2 - 1, y2)] = None

        if piece[1] == 'K' and abs(y2 - y1) == 2:
            if y1 < y2:  # Queen-side castling
                rook_start = (x1, 8)
                rook_end = (x1, y2 - 1)
            else:  # Queen-side castling
                rook_start = (x1, 1)
                rook_end = (x1, y2 + 1)


            rook_piece = board.get(rook_start)
            board[rook_start] = None
            board[rook_end] = rook_piece

        if piece[1] == 'K':
            new_castling_rights[piece[0]]['Q'] = False
            new_castling_rights[piece[0]]['K'] = False
        
        if piece[1] == 'R' and y1 == 1:
            new_castling_rights[piece[0]]['K'] = False
        
        if piece[1] == 'R' and y1 == 8:
            new_castling_rights[piece[0]]['Q'] = False

        if piece[1] == 'P' and abs(x2-x1) == 2:
            if piece[0] == 'W':
                new_en_passant_square = (x2-1,y2)
            else:
                new_en_passant_square = (x2+1,y2)
        
        next_player = 'B' if state.to_move == 'W' else 'W'
        moves_last_capture += 1
        moves = []
        for (x, y), piece in board.items():
            if piece is not None and piece[0] == next_player:
                possible_moves = self.get_moves(board, (x, y), piece, castling_rights=new_castling_rights, en_passant_square=new_en_passant_square)
                if possible_moves:
                    moves.append(((x, y), possible_moves))

        return GameState(to_move=next_player, utility=state.utility, board=board, moves=moves, castling_rights=new_castling_rights, en_passant_square=new_en_passant_square, moves_last_capture=moves_last_capture)



    def utility(self, state, player):
        """Returns 1 for black winning, -1 for player white winning and 0 for a draw."""
        
        if self.is_stalemate(state.board, player) or state.moves_last_capture >= 50:
            return 0
        
        return 1 if player == 'B' else -1

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        # Check if the current player has any legal moves
        if state.moves_last_capture >= 50:
            return True
        
        current_player = state.to_move

        for (x, y), piece in state.board.items():
            if piece and current_player == piece[0]:
                possible_moves = self.get_moves(state.board, (x, y), piece, state.castling_rights, state.en_passant_square)
                if possible_moves:
                    return False

        # Game over if no moves are left
        return True



    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):

        """Display the board to the terminal."""

        piece_symbols = {
            "WP": "♙", "WR": "♖", "WN": "♘", "WB": "♗", "WQ": "♕", "WK": "♔",
            "BP": "♟", "BR": "♜", "BN": "♞", "BB": "♝", "BQ": "♛", "BK": "♚"
        }
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
             
                if piece in piece_symbols:
                    row_display.append(piece_symbols[piece])
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
   
        ai_move = ai_player_fn(self, state)
        state = self.result(state, ai_move)

        return state, ai_move


    def play_game(self, *players):
        """Play a 2-person, move-alternating game."""
        state = self.initial

        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
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

def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):

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
    if current_player == "W":
        symbol = "black"
    else:
        symbol = "white"
    move_options = []
    print(f"\nAvailable moves for {symbol}:")
    move_index = 1

    # Get all possible moves for the current player
    for move in game.actions(state):
        piece_position, target_position = move
        piece = state.board[piece_position]
        if piece[0] == current_player:
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


# Random player
def random_player(game, state):
    """Returns a random move for the current player in chess."""
    current_player = state.to_move

    valid_moves = [
        move for move in game.actions(state)
        if state.board.get(move[0], '').startswith(current_player)
    ]

    return random.choice(valid_moves) if valid_moves else None


def medium(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=evaluation.evaluate_board)

def hard(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=evaluation.evaluate_board_with_check)

def impossible(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=evaluation.evaluate_board_with_check)

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

        if current_player == "W":
            action = player_x(game, state)
        else:
            action = player_o(game, state)

        move_end_time = time.time()

        # Update thinking time for each player
        if current_player == "W":
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
        # ("Random Player X", random_player),
        # ("Count Complete X d2", count_complete_X_d1),
        # ("Count Complete X d2", count_complete_X_d2),
        # ("Count Pieces X d1", count_pieces_X_d1),
        # ("Count Pieces X d2", count_pieces_X_d2),
        # ("Count Pieces with King X d1", count_pieces_with_king_X_d1),
        # ("Count Pieces with King X d2", count_pieces_with_king_X_d2),
        # ("Count Simple Center X d1", count_simple_center_X_d1),
        # ("Count Simple Center X d2", count_simple_center_X_d2),
        # ("Count King Center X d1", count_king_center_X_d1),
        # ("Count King Center X d2", count_king_center_X_d2),
        # ("Count Simple Cluster X d1", count_simple_cluster_X_d1),
        # ("Count Simple Cluster X d2", count_simple_cluster_X_d2),
        # ("Count King Cluster X d1", count_king_cluster_X_d1),
        # ("Count King Cluster X d2", count_king_cluster_X_d2),
        # ("Count Simple Complete X d1", count_simple_complete_X_d1),
        # ("Count Simple Complete X d2", count_simple_complete_X_d2),
        ("Count Pieces White d3", count_pieces_white_d3)
    ]
    o_players = [
        # ("Count Complete O d1", count_complete_O_d1),
        # ("Count Complete O d2", count_complete_O_d2),
        # # ("Count Complete O d3", count_complete_O_d3),
        # # ("Count Complete O d4", count_complete_O_d4),
        # # ("Count Complete O d5", count_complete_O_d5),
        # # ("Count Complete O d6", count_complete_O_d6),
        # ("Count Simple Complete O d1", count_simple_complete_O_d1),
        # ("Count Simple Complete O d2", count_simple_complete_O_d2),
        # ("Count King Cluster O d1", count_king_cluster_O_d1),
        # ("Count King Cluster O d2", count_king_cluster_O_d2),
        # ("Count Simple Cluster O d1", count_simple_cluster_O_d1),
        # ("Count Simple Cluster O d2", count_simple_cluster_O_d2),
        # ("Count King Center O d1", count_king_center_O_d1),
        # ("Count King Center O d2", count_king_center_O_d2),
        # ("Count Simple Center O d1", count_simple_center_O_d1),
        # ("Count Simple Center O d2", count_simple_center_O_d2),
        # ("Count Pieces with King O d1", count_pieces_with_king_O_d1),
        # ("Count Pieces with King O d2", count_pieces_with_king_O_d2),
        # # ("Count Pieces with King O d3", count_pieces_with_king_O_d3),
        # # ("Count Pieces with King O d4", count_pieces_with_king_O_d4),
        # # ("Count Pieces with King O d5", count_pieces_with_king_O_d5),
        # # ("Count Pieces with King O d6", count_pieces_with_king_O_d6),
        # ("Count Pieces O d1", count_pieces_O_d1),
        # ("Count Pieces O d2", count_pieces_O_d2),
        ("Random Player Black", random_player),
    ]
    game_class = Chess

    print("Welcome! Please choose an option:")
    print("1. Play against an AI")
    print("2. Run a tournament")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # User chooses to play against an AI
        print("Choose AI difficulty:")
        print("1. Beginner")
        print("2. Intermediate")
        # print("3. Advanced")
        # print("4. Expert")

        difficulty = input("Enter the difficulty level (1, 2, 3, or 4): ").strip()
        if difficulty == "1":
            ai_player = random_player
            print("You selected Beginner AI.")
        elif difficulty == "2":
            ai_player = count_pieces_white_d3
            print("You selected Intermediate AI.")
        # elif difficulty == "3":
        #     ai_player = count_pieces_with_king_X_d4
        #     print("You selected Advanced AI.")
        # elif difficulty == "4":
        #     ai_player = count_complete_X_d6
        #     print("You selected Expert AI.")
        else:
            print("Invalid choice. Defaulting to Beginner AI.")
            ai_player = random_player

        # Start a game with the selected difficulty
        chess = game_class()
        result = chess.play_game(ai_player, query_player)

        if result > 0:
            print("White wins!")
        elif result < 0:
            print("Black wins!")
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