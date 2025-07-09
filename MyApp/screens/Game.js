import { View, Text, Pressable } from 'react-native';
import React, { useEffect, useState } from 'react';

export default function Game({ route, navigation }) {
  const difficulty = route?.params?.difficulty || 'Easy';
  const [board, setBoard] = useState(null);
  const [moves, setMoves] = useState([]);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [validMoves, setValidMoves] = useState([]);

  useEffect(() => {
    const startNewGame = async () => {
      try {
        console.log(difficulty);
        const response = await fetch('http://localhost:5000/api/newGame', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ difficulty }),
        });

        const data = await response.json();
        console.log('Game Started:', data);

        setBoard(data.board);
        setMoves(data.moves);
        console.log("Moves: ", data.moves);
      } catch (error) {
        console.error('Error starting game:', error);
      }
    };

    startNewGame();

    return () => {
      console.log('GameScreen unmounted');
    };
  }, []);


  const handleSquarePress = (row, col) => {
    const piece = board[row][col];
    const isPlayer1Piece = piece === 'X' || piece === 'KX';

    const clickedSquare = [row + 1, col + 1];

    const isValidMove = validMoves.some(
            ([x, y]) => x === clickedSquare[0] && y === clickedSquare[1]
        );

        if (isValidMove && selectedPiece) {
            const makeMove = async () => {
            try {
                const response = await fetch('http://localhost:5000/api/makeMove', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        from: selectedPiece,
                        to: clickedSquare,
                        board: board,
                        difficulty: difficulty,
                        player: 'O'
                    }),
                });

                const data = await response.json();
                console.log('Move made:', data);

                setBoard(data.board);
                setMoves(data.moves);
                const won = data.won == -1;
                if (data.game_over) {
                    navigation.navigate('GameOver', { won: won});
                }
                setSelectedPiece(null);
                setValidMoves([]);
            } catch (error) {
                console.error('Error making move:', error);
            }
            };

            makeMove();
        } else {
            
            const moveData = moves.find(
                (m) => m.from[0] === clickedSquare[0] && m.from[1] === clickedSquare[1]
            );

            if (moveData) {
                setSelectedPiece(moveData.from);
                setValidMoves(moveData.to);
            } else {
                setSelectedPiece(null);
                setValidMoves([]);
            }
        }
    };


  const renderBoard = () => {
    const boardView = [];

    for (let row = 0; row < 8; row++) {
      const rowView = [];

      for (let col = 0; col < 8; col++) {
        const isDark = (row + col) % 2 === 1;
        const piece = board?.[row]?.[col];
        const isPlayer1Piece = piece === 'X' || piece === 'KX';
        const isPlayer2Piece = piece === 'O' || piece === 'KO';

        const isSelected =
          selectedPiece &&
          selectedPiece[0] === row + 1 &&
          selectedPiece[1] === col + 1;

        const isValidMove = validMoves.some(
          ([x, y]) => x === row + 1 && y === col + 1
        );

        rowView.push(
          <Pressable
            key={`${row}-${col}`}
            onPress={() => handleSquarePress(row, col)}
            className={`w-10 h-10 ${isDark ? 'bg-gray-900' : 'bg-gray-400'} justify-center items-center border ${isSelected ? 'border-2 border-yellow-500' : 'border-transparent'}`}
            >
            {isPlayer1Piece && (
                <View className="w-6 h-6 bg-red-500 rounded-full items-center justify-center">
                {piece === 'KX' && (
                    <Text className="text-white text-xs font-bold">K</Text>
                )}
                </View>
            )}
            {isPlayer2Piece && (
                <View className="w-6 h-6 bg-black rounded-full items-center justify-center">
                {piece === 'KO' && (
                    <Text className="text-white text-xs font-bold">K</Text>
                )}
                </View>
            )}
            {isValidMove && (
                <View className="w-3 h-3 bg-gray-600 opacity-80 rounded-full absolute" />
            )}
            </Pressable>

        );
      }

      boardView.push(
        <View key={row} className="flex-row">
          {rowView}
        </View>
      );
    }

    return boardView;
  };

  return (
    <View className="flex-1 bg-gray-700 items-center justify-center">
      <Text className="text-xl font-bold text-slate-300 mb-8">Difficulty: {difficulty}</Text>
      <View className="border-0 border-green-600">
        {board ? renderBoard() : <Text className="text-slate-300">Loading board...</Text>}
      </View>
    </View>
  );
}
