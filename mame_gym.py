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

        # Connect to a MAME instance launched with e.g "mame -autoboot_script mame_server.lua"
        self.mame.connect()
        
        # Make sure the game is prepped from the start 
        self.reset()

        # Fetch relevant values 
        self.width, self.height = self._get_screen_size()
        self.bytes_len = self._get_pixels_bytes()

        # Define action and observation spaces
        # This is an example for Joust, adjust as needed for other games
        self.action_space = spaces.Discrete(6)  # Left, Right, Flap, Left+Flap, Right+Flap, No-op
        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8) 

    def step(self, action):
        self._send_action(action)
        self._step()
        observation = self._get_observation()
        reward = self._calculate_reward(player=1)
        done = self._check_done(player=1)
        info = {}
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset the mame emulation
        self._soft_reset()
        
        # initialize last score and lives for both players
        self.last_score = {1: self._get_score(1), 2: self._get_score(2)}
        self.last_lives = {1: self._get_lives(1), 2: self._get_lives(2)}
        
        self._get_ready_to_play()

        # get initial observation
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

    def _get_ready_to_play(self):
        # Insert a coin
        self.mame.execute("manager.machine.input.port_by_tag('COIN1').fields['COIN1'].set_value(1)")
        # Press start button
        self.mame.execute("manager.machine.input.port_by_tag('START').fields['START'].set_value(1)")

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
        P1_ACTIONS = [ "P1_LEFT", "P1_RIGHT", "P1_BUTTON1", "P1_LEFT P1_BUTTON1", "P1_RIGHT P1_BUTTON1", ""]
        P2_ACTIONS = [ "P2_LEFT", "P2_RIGHT", "P2_BUTTON1", "P2_LEFT P2_BUTTON1", "P2_RIGHT P2_BUTTON1", ""]
        actions = P1_ACTIONS if player == 1 else P2_ACTIONS
        mame_action = actions[action]
        if mame_action: self._send_input(mame_action)

    
    # Memory addresses for Joust
    LIVES_ADDR = {1: 0xA052, 2: 0xA05C}
    SCORE_ADDR = {1: 0xA04C, 2: 0xA058}
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. TODO: End of stage bonus?
    
    def _read_byte(self, address):
        result = self.mame.execute(f"return manager.machine.devices[':maincpu'].spaces['program']:read_u8(0x{address:X})")
        return int(result.decode())

    def _read_word(self, address):
        result = self.mame.execute(f"return manager.machine.devices[':maincpu'].spaces['program']:read_u16(0x{address:X})")
        return int(result.decode())

    def _get_lives(self, player=1):
        return self._read_byte(self.LIVES_ADDR[player])

    def _get_score(self, player=1):
        return self._read_word(self.SCORE_ADDR[player])

    def _calculate_reward(self, player=1):
        # Get current score and lives
        current_score = self._get_score(player)
        current_lives = self._get_lives(player)
        
        # Calculate score,lives differences
        score_diff = current_score - self.last_score[player]
        lives_diff = current_lives - self.last_lives[player]
        
        # Update last score and lives
        self.last_score[player] = current_score
        self.last_lives[player] = current_lives
        
        # Reward for score increase
        reward = score_diff / self.MAX_SCORE_DIFF  # Normalize score difference
        
        # Penalty for losing lives
        if lives_diff < 0:
            reward -= 1.0  # Penalty for each life lost
        
        return reward

    def _check_done(self, player=1):
        current_lives = self._get_lives(player)
        return current_lives == 0  # Game is over when player has no lives left


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