import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import MainScreen from './screens/MainScreen';
import Game from './screens/Game';
import GameOver from './screens/GameOver';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Main">
        <Stack.Screen name="Main" component={MainScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Game" component={Game} options={{ headerShown: false }}/>
        <Stack.Screen name="GameOver" component={GameOver} options={{ headerShown: false }}/>
      </Stack.Navigator>
    </NavigationContainer>
  );
}
