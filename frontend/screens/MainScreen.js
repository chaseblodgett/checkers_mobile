import { Text, TouchableOpacity, View } from 'react-native';

export default function MainScreen({ navigation }) {
  return (
    <View className="flex-1 bg-gray-700 justify-center items-center px-6">
      <Text className="text-slate-300 text-5xl font-bold mb-8 text-center">
        Checkers.ai
      </Text>
      <Text className="text-slate-300 text-3xl mb-8 text-center">
        Choose a difficulty.
      </Text>

      {['Easy', 'Medium', 'Hard', 'Impossible'].map((level) => (
        <TouchableOpacity
          key={level}
          className="w-full py-4 bg-gray-400 border-2 border-green-600 rounded-2xl mb-4 shadow-md active:opacity-80"
          onPress={() => navigation.navigate('Game', { difficulty: level })}
        >
          <Text className="text-slate-900 text-center text-lg font-semibold">{level}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}
