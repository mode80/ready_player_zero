from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mame_client import MAMEClient
import matplotlib.pyplot as plt # type:ignore

class MAMEEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, game='joust', render_mode=None):
        super().__init__()
        self.mame = MAMEClient()
        self.game = game
        self.render_mode = render_mode

        # Connect to a MAME instance launched with -autoboot_script mame_server.lua  
        self.mame.connect()
        
        # Make sure the game is prepped from the start 
        self._soft_reset()

        # Fetch relevant values 
        self.width, self.height = self._get_screen_size()
        self.bytes_len = self._get_pixels_bytes()

        # Define action and observation spaces
        # This is an example for Joust, adjust as needed for other games
        self.action_space = spaces.Discrete(6)  # Left, Right, Flap, Left+Flap, Right+Flap, No-op
        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8) 

    def step(self, action):
        # Send action to MAME
        self._send_action(action)

        # Step the emulation forward
        self._step()

        # Get the new observation
        observation = self._get_observation()

        # Calculate reward (you'll need to implement game-specific reward logic)
        reward = self._calculate_reward()

        # Check if the episode is done
        done = self._check_done()

        # Get additional info
        info = {}

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the MAME emulation
        self._soft_reset()

        # Get initial observation
        observation = self._get_observation()

        info = {}
        return observation, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_observation()
        else:
            return None

    def close(self):
        self.mame.close()

    def _pause(self):
        self.mame.execute("emu.pause()")

    def _unpause(self):
        self.mame.execute("emu.unpause()")

    def _throttled_on(self):
        self.mame.execute("manager.machine.video.throttled = true")

    def _throttled_off(self):
        self.mame.execute("manager.machine.video.throttled = false")

    def _step(self):
        self.mame.execute("emu.step()")

    def _soft_reset(self):
        self._unpause()
        self.mame.execute("manager.machine.soft_reset()")
        self._pause()
        self._throttled_off()

    def _get_screen_size(self):
        result = self.mame.execute("s=manager.machine.screens[':screen']; return s.width .. 'x' .. s.height")
        width, height = map(int, result.decode().split('x'))
        return width, height

    def _get_pixels_bytes(self):
        result = self.mame.execute("s=manager.machine.screens[':screen']; return #s:pixels()")
        return int(result.decode())

    def _get_frame_number(self):
        result = self.mame.execute("s=manager.machine.screens[':screen']; return s:frame_number()")
        return int(result.decode())

    def _get_pixels(self):
        return self.mame.execute("s=manager.machine.screens[':screen']; return s:pixels()")

    def _send_input(self, input_command):
        lua_code = f"""
        local button = manager.machine.input:code_from_token('P1_{input_command}');
        manager.machine.input:code_pressed(button);
        manager.machine.input:code_released(button);
        """
        self.mame.execute(lua_code)

    def _get_observation(self):
        pixels = self._get_pixels()
        # trim any "footer" data beyond pixel values of self.height * self.width * 4
        pixels = pixels[:self.height * self.width * 4]
        # unflatten bytes into row,col,channel format; keep all rows and cols, but transform 'ABGR' to RGB  
        observation = np.frombuffer(pixels[:self.height * self.width * 4], 
                                    dtype=np.uint8).reshape((self.height, self.width, 4))[:,:,2::-1]
        return observation

    def _send_action(self, action, player=1):
        # Map action to MAME input commands
        player1_actions = [ "P1_LEFT", "P1_RIGHT", "P1_BUTTON1", "P1_LEFT P1_BUTTON1", "P1_RIGHT P1_BUTTON1", ""]
        player2_actions = [ "P2_LEFT", "P2_RIGHT", "P2_BUTTON1", "P2_LEFT P2_BUTTON1", "P2_RIGHT P2_BUTTON1", ""]
        actions = player1_actions if player == 1 else player2_actions
        mame_action = actions[action]
        if mame_action: self._send_input(mame_action)

    def _calculate_reward(self):
        # Implement game-specific reward logic
        # For Joust, this could be based on score, surviving, defeating enemies, etc.
        # You'll need to read game memory to get this information
        return 0

    def _check_done(self):
        # Implement game-specific logic to check if the episode is done
        # This could be based on lives remaining, game over flag, etc.
        return False

# Example usage
if __name__ == "__main__":
    env = MAMEEnv(game='joust', render_mode='rgb_array')

    observation, info = env.reset()

    # # debug display the observation as an image
    # env._unpause()
    # sleep(6)
    # plt.imshow( env._get_observation() )

    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            observation, info = env.reset()

    env.close()


def sample_PPO():
    """
    from stable_baselines3 import PPO

    env = MAMEEnv(game='joust')
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Test the trained model
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    """
    pass