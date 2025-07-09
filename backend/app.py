from flask import Flask, request, jsonify
from flask_cors import CORS
import checkers

app = Flask(__name__)
CORS(app)


ai_bots = {
    'Easy' : checkers.random_player,
    'Medium' : checkers.count_pieces_X_d2,
    'Hard' : checkers.count_pieces_with_king_X_d4,
    'Impossible' : checkers.count_complete_X_d6
}

@app.route('/api/hello')
def hello():
    return jsonify(message="Hello from Flask!")


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

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
