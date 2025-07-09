import React from 'react';
import { View, Text, Pressable } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';

export default function GameOver({ route, navigation }) {
  const won = route?.params?.won || false;

  const handlePlayAgain = () => {
    navigation.navigate('Main'); 
  };

  return (
    <View className="flex-1 bg-gray-800 items-center justify-center px-4">
      <Text className="text-3xl font-bold text-white mb-6">
        {won ? '🎉 You Won!' : '😞 You Lost'}
      </Text>

      <Pressable
        onPress={handlePlayAgain}
        className="bg-green-600 px-6 py-3 rounded-2xl shadow-md"
      >
        <Text className="text-white text-lg font-semibold">Play Again</Text>
      </Pressable>
    </View>
  );
}
