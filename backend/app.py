from flask import Flask, request, jsonify
from flask_cors import CORS
import checkers
import chess

app = Flask(__name__)
CORS(app)


ai_bots = {
    'Easy' : checkers.random_player,
    'Medium' : checkers.count_pieces_X_d2,
    'Hard' : checkers.count_pieces_with_king_X_d4,
    'Impossible' : checkers.count_complete_X_d6
}

chess_ai_bots = {
    'Easy' : chess.random_player,
    'Medium' : chess.count_pieces_white_d3
}

@app.route('/api/makeMove', methods=['POST'])
def make_move():
    data = request.get_json()
    board_list = data.get('board', [])
    end = data.get('to', [])
    start = data.get('from', [])
    player = data.get('player', "") 
    ai_bot = data.get('difficulty', "")

    move = ((start[0], start[1]), (end[0], end[1]))

    board_dict = {}
    for x in range(8):
        for y in range(8):
            piece = board_list[x][y]
            board_dict[(x + 1, y + 1)] = piece
            
  
    game = checkers.Checkers(to_move=player, board=board_dict)
    state = game.initial

    def ai_player_fn(game, state): return ai_bots[ai_bot](game, state)

    new_state, ai_move = game.make_turn(state, move, ai_player_fn)

    board = new_state.board
    board_array = [[None for _ in range(8)] for _ in range(8)]
    for (x, y), value in board.items():
        board_array[x - 1][y - 1] = value

    json_moves = [
        {
            "from": [x, y],
            "to": [[x2, y2] for (x2, y2) in possible_moves]
        }
        for ((x, y), possible_moves) in new_state.moves
    ]

    return jsonify({
        'board': board_array,
        'to_move': game.to_move(new_state),
        'moves' : json_moves,
        'ai_move': ai_move,
        'game_over': game.terminal_test(new_state),
        'won' : game.utility(new_state, game.to_move(state))
    }), 200

@app.route('/api/newGame', methods=['POST'])
def new_game():
    data = request.get_json()
    difficulty = data.get('difficulty', '')
    to_move = 'X'

    def ai_player_fn(game, state): return ai_bots[difficulty](game, state)

    game = checkers.Checkers()
    state = game.initial

    new_state, ai_move = game.make_turn(state, move=None, ai_player_fn=ai_player_fn)

    board = new_state.board
    board_array = [[None for _ in range(8)] for _ in range(8)]
    for (x, y), value in board.items():
        board_array[x - 1][y - 1] = value


    json_moves = [
        {
            "from": [x, y],
            "to": [[x2, y2] for (x2, y2) in possible_moves]
        }
        for ((x, y), possible_moves) in new_state.moves
    ]

    return jsonify({
        'board': board_array,
        'to_move': game.to_move(new_state),
        'moves' : json_moves,
        'ai_move': ai_move,
        'game_over': game.terminal_test(new_state)

    }), 200

@app.route('/api/makeChessMove', methods=['POST'])
def make_chess_move():
    data = request.get_json()
    board_list = data.get('board', [])
    end = data.get('to', [])
    start = data.get('from', [])
    player = data.get('player', "") 
    ai_bot = data.get('difficulty', "")
    castling_rights = data.get('castlingRights', {})
    en_passant_square = data.get('enPassantSquare', [])
    moves_last_capture = data.get('movesLastCapture', 0)

    if en_passant_square is not None:
        en_passant_square = tuple(en_passant_square)

    move = ((start[0], start[1]), (end[0], end[1]))

    board_dict = {}
    for x in range(8):
        for y in range(8):
            piece = board_list[x][y]
            board_dict[(x + 1, y + 1)] = piece
            
  
    game = chess.Chess(to_move=player, board=board_dict, castling_rights=castling_rights, en_passant_square=en_passant_square, moves_last_capture=moves_last_capture)
    state = game.initial

    def ai_player_fn(game, state): return chess_ai_bots[ai_bot](game, state)

    new_state, ai_move = game.make_turn(state, move, ai_player_fn)

    board = new_state.board
    board_array = [[None for _ in range(8)] for _ in range(8)]
    for (x, y), value in board.items():
        board_array[x - 1][y - 1] = value

    json_moves = [
        {
            "from": [x, y],
            "to": [[x2, y2] for (x2, y2) in possible_moves]
        }
        for ((x, y), possible_moves) in new_state.moves
    ]
    print(new_state.castling_rights)
    print(new_state.moves_last_capture)
    return jsonify({
        'board': board_array,
        'to_move': game.to_move(new_state),
        'moves' : json_moves,
        'ai_move': ai_move,
        'game_over': game.terminal_test(new_state),
        'won' : game.utility(new_state, game.to_move(new_state)),
        'castling_rights' : new_state.castling_rights,
        'en_passant_square' : new_state.en_passant_square,
        'moves_last_capture' : new_state.moves_last_capture
    }), 200

@app.route('/api/newChessGame', methods=['POST'])
def new_chess_game():
    data = request.get_json()
    difficulty = data.get('difficulty', '')

    def ai_player_fn(game, state): return chess_ai_bots[difficulty](game, state)

    game = chess.Chess()
    state = game.initial

    new_state, ai_move = game.make_turn(state, move=None, ai_player_fn=ai_player_fn)

    board = new_state.board
    board_array = [[None for _ in range(8)] for _ in range(8)]
    for (x, y), value in board.items():
        board_array[x - 1][y - 1] = value


    json_moves = [
        {
            "from": [x, y],
            "to": [[x2, y2] for (x2, y2) in possible_moves]
        }
        for ((x, y), possible_moves) in new_state.moves
    ]
    
    return jsonify({
        'board': board_array,
        'to_move': game.to_move(new_state),
        'moves' : json_moves,
        'ai_move': ai_move,
        'game_over': game.terminal_test(new_state),
        'castling_rights' : new_state.castling_rights,
        'en_passant_square' : new_state.en_passant_square,
        'moves_last_capture' : new_state.moves_last_capture
    }), 200

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
