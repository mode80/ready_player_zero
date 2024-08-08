import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mame_client import MAMEConsole

class MAMEEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, game='joust', render_mode=None):
        super().__init__()
        self.mame = MAMEConsole()
        self.game = game
        self.render_mode = render_mode

        # Connect to MAME
        self.mame.connect()

        # Get screen dimensions
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
        self.mame.execute("emu.step()")

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
        self.mame.execute("emu.soft_reset()")

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

    def _get_screen_size(self):
        result = self.mame.execute("return string.pack('>II', screen.width, screen.height)")
        return struct.unpack('>II', result)

    def _get_pixels_bytes(self):
        result = self.mame.execute("return string.pack('>I', #screen:pixels())")
        return struct.unpack('>I', result)[0]

    def _get_frame_number(self):
        result = self.mame.execute("return string.pack('>I', screen:frame_number())")
        return struct.unpack('>I', result)[0]

    def _get_pixels(self):
        return self.mame.execute("return screen:pixels()")

    def _send_input(self, input_command):
        lua_code = f"""
        local field = ioport.ports[':IN0']:field('P1 {input_command}')
        field:set_value(1)
        emu.wait_time(0.016)  -- Wait for approximately one frame (assuming 60 FPS)
        field:set_value(0)
        """
        self.mame.execute(lua_code)

    def _get_observation(self):
        pixels = self._get_pixels()
        observation = np.frombuffer(pixels, dtype=np.uint8).reshape((self.height, self.width, 4))
        # Convert RGBA to RGB by discarding the alpha channel
        observation = observation[:,:,:3]
        return observation

    def _send_action(self, action):
        # Map action to MAME input commands
        actions = [
            "LEFT", "RIGHT", "BUTTON1",
            "LEFT BUTTON1", "RIGHT BUTTON1", ""  # No-op
        ]
        mame_action = actions[action]
        if mame_action:
            self._send_input(mame_action)

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