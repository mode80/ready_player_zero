import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from libretro import SessionBuilder, DefaultPathDriver, SubsystemContent, ContentDriver
from libretro.h import RETRO_MEMORY_SYSTEM_RAM  # Adjust the import based on your libretro.h path
from libretro.drivers import ArrayAudioDriver, GeneratorInputDriver, ArrayVideoDriver, DictOptionDriver

class JoustEnv(gym.Env):
    """
    Gym environment for the classic arcade game Joust using libretro.py.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(JoustEnv, self).__init__()

        self.render_mode = render_mode
        if self.render_mode == 'human':
            pygame.init()
            self.screen = None

        # Define action space: Example for Joust (8 discrete actions)
        # Adjust the number of actions based on actual game controls
        self.action_space = spaces.Discrete(8)

        # Define observation space: Assuming frames are 224x256 RGB images
        # Adjust the shape and dtype as per actual frame size and format
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(224, 256, 3),
            dtype=np.uint8
        )

        # Initialize libretro session with configured drivers
        self.session = (
            SessionBuilder()
            .with_core('/Users/user/Library/Application Support/RetroArch/cores/mame2003_plus_libretro.dylib')
            .with_content('/Users/user/mame/roms/joust.zip')
            .with_audio(ArrayAudioDriver())
            .with_input(GeneratorInputDriver())
            .with_video(ArrayVideoDriver())
            .with_paths(
                DefaultPathDriver(
                    corepath='/Users/user/Library/Application Support/RetroArch/cores/mame2003_plus_libretro.dylib',
                    system='/Users/user/Library/CloudStorage/Dropbox/code/ready_player_zero/system',
                    assets='/Users/user/Library/CloudStorage/Dropbox/code/ready_player_zero/assets',
                    save='/Users/user/Library/CloudStorage/Dropbox/code/ready_player_zero/save',
                    playlist='/Users/user/Library/CloudStorage/Dropbox/code/ready_player_zero/playlist',
                )
            )
            .with_options({
                b'melonds_option1': b'value1',
                b'melonds_option2': b'value2',
                # Add other necessary options here
            })
            .build()
        )

        self.session.__enter__()

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Map Gym action to emulator input
        self._send_action(action)

        # Run one frame of the emulator
        self.session.run()

        # Capture the current frame
        frame = self._get_frame()

        # Extract state information
        score = self._get_score()
        lives = self._get_lives()
        done = self._check_done()

        # Define reward: Example based on score
        reward = score  # Modify as needed for your RL objectives

        # Additional info
        info = {'score': score, 'lives': lives}

        return frame, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.session.reset()
        initial_frame = self._get_frame()
        return initial_frame

    def render(self, mode='human'):
        """
        Render the environment.
        """
        frame = self._get_frame()
        if self.render_mode == 'human':
            if self.screen is None:
                self.screen = pygame.display.set_mode(
                    (frame.shape[1], frame.shape[0])
                )
            pygame.surfarray.blit_array(self.screen, frame)
            pygame.display.flip()
        elif mode == 'rgb_array':
            return frame

    def close(self):
        """
        Perform any necessary cleanup.
        """
        if self.render_mode == 'human':
            pygame.quit()
        self.session.__exit__(None, None, None)

    def _send_action(self, action):
        """
        Convert and send the discrete action to the emulator.
        """
        # Define mapping from actions to emulator inputs
        # Example mapping for 8 actions
        input_state = np.zeros(12, dtype=np.uint32)  # Example: 12 possible buttons
        if action == 0:
            input_state[0] = 1  # Example: Move Left
        elif action == 1:
            input_state[1] = 1  # Example: Move Right
        elif action == 2:
            input_state[2] = 1  # Example: Jump
        elif action == 3:
            input_state[3] = 1  # Example: Attack
        elif action == 4:
            input_state[4] = 1  # Example: Special
        elif action == 5:
            input_state[5] = 1  # Example: Pause
        elif action == 6:
            input_state[6] = 1  # Example: Menu
        elif action == 7:
            input_state[7] = 1  # Example: Select
        # Add additional action mappings as needed
        self.session.input.set_state(input_state)

    def _get_frame(self):
        """
        Capture the current video frame from the emulator.
        """
        screenshot = self.session.video.screenshot()
        frame = np.array(screenshot.data, dtype=np.uint8)
        frame = frame.reshape((screenshot.height, screenshot.width, 3))
        return frame

    def _get_score(self):
        """
        Retrieve the current score from the emulator's memory.
        """
        memory = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        if memory is None:
            return 0
        # Example: Assuming score is stored at memory address 0x00 to 0x04
        score = int.from_bytes(memory[0x00:0x04], byteorder='little')
        return score

    def _get_lives(self):
        """
        Retrieve the current number of lives from the emulator's memory.
        """
        memory = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        if memory is None:
            return 0
        # Example: Assuming lives are stored at memory address 0x04
        lives = int.from_bytes(memory[0x04:0x05], byteorder='little')
        return lives

    def _check_done(self):
        """
        Determine if the game has ended.
        """
        lives = self._get_lives()
        return lives <= 0

    def _compute_reward(self, score, lives):
        """
        Compute the reward based on the current state.
        """
        # Example: Reward based solely on score
        return score