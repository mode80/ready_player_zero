import gymnasium as gym
import numpy as np
import pygame
import libretro
from libretro import SessionBuilder, DefaultPathDriver, SubsystemContent, ContentDriver, JoypadState
from libretro.h import RETRO_MEMORY_SYSTEM_RAM  
from libretro.drivers import ArrayAudioDriver, GeneratorInputDriver, ArrayVideoDriver, DictOptionDriver, UnformattedLogDriver
import tempfile, os 
import logging 
import skimage
from itertools import repeat

class JoustEnv(gym.Env):
    """ Gym environment for the classic arcade game Joust using libretro.py.  """

    DOWNSCALE = 1 # Downscale the image by this factor (> 1 to speed up training)
    FRAMES_PER_STEP = 5     # 12 press-or-relase actions (6 complete button presses) per second is reasonable for human reflexes 
                            # Joust might react based on a count of input flags over the last [?] frames 
                            # but best way to handle this is probably to feed it a history of [?] previous inputs in the observation

    WIDTH, HEIGHT = 292, 240#146#240 # pixel dimensions of the screen for this rom
    # CORE_PATH= '/Users/user/Library/Application Support/RetroArch/cores/fbneo_libretro.dylib'
    # CORE_PATH= '/Users/user/Library/Application Support/RetroArch/cores/mame2000_libretro.dylib'
    CORE_PATH= '/Users/user/Library/Application Support/RetroArch/cores/mame2003_plus_libretro.dylib'
    ROM_PATH= '/Users/user/mame/roms/joust.zip'
    ROM_PATH= '/Users/user/Documents/RetroArch/fbneo/roms/arcade/joust.zip'
    SYSTEM_PATH= '/Users/user/Documents/RetroArch/system'
    ASSETS_PATH= '/Users/user/Documents/RetroArch/assets'
    SAVE_PATH= '/Users/user/Documents/RetroArch/saves'
    PLAYLIST_PATH= '/Users/user/Documents/RetroArch/playlists'
     # MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. 
    BOOT_FRAMES = 700 
    READY_UP_FRAMES = 225

    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
 
    NOOP, LEFT, RIGHT = JoypadState(), JoypadState(left=True), JoypadState(right=True)
    FLAP, FLAP_LEFT, FLAP_RIGHT, = JoypadState(b=True), JoypadState(b=True, left=True), JoypadState(b=True, right=True)
    COIN, START = JoypadState(select=True), JoypadState(start=True)
    

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(JoustEnv, self).__init__()

        # Define action space: Example for Joust 
        # Adjust the number of actions based on actual game controls
        # self.actions = [self.NOOP, self.FLAP]#, self.FLAP_LEFT, self.FLAP_RIGHT]
        self.actions = [self.NOOP, self.FLAP, self.LEFT, self.RIGHT, self.FLAP_LEFT, self.FLAP_RIGHT]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, self.HEIGHT, self.WIDTH), 
            dtype=np.uint8
        )

        self.log_driver = UnformattedLogDriver()
        self.log_driver._logger.setLevel(logging.WARN)
        self.p1_input = JoypadState()

        def input_callback(): 
            i=0
            # Bypass start sequence
            yield from repeat(0, self.BOOT_FRAMES)
            yield from repeat(self.COIN, 2) 
            yield from repeat(self.START, 8) 
            yield from repeat(0, self.READY_UP_FRAMES)
            # take input during play
            while True:
                # i=i+1; print(i, self.p1_input)
                yield self.p1_input

        self.last_score, self.last_lives = 0,0 

        # for rendering
        self.render_mode = render_mode
        self.pixel_history = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Initialize libretro session with configured drivers
        temp_dir = tempfile.TemporaryDirectory(".libretro").name
        self.session = (
            libretro
            .defaults(self.CORE_PATH)
            .with_content(self.ROM_PATH)
            .with_input(GeneratorInputDriver(input_callback))
            .with_log(self.log_driver)
            .with_paths(
                DefaultPathDriver(
                    corepath = self.CORE_PATH,
                    system = self.SYSTEM_PATH,
                    assets = self.ASSETS_PATH,
                    save = self.SAVE_PATH,
                    playlist = self.PLAYLIST_PATH,
                )
            )
            .build()
        )

        self.session.__enter__()

    def step(self, action):
        """ Execute one time step within the environment.  """

        # Run frame(s) of the emulator
        self._set_action(action)
        for _ in range(self.FRAMES_PER_STEP): self.session.run()

        # Capture the current frame
        pixels = self._get_frame()
        
        # 3-frame history
        self.pixel_history = np.roll(self.pixel_history, 1, axis=0)  # cycle 'channel' from [old, older, oldest] to [oldest, old, older]
        self.pixel_history[0] = pixels # replace oldest with current observation

        # Extract state information
        score = self._get_score()
        lives = self._get_lives()

        # Define reward: Example based on score
        reward = score  # Modify as needed for your RL objectives

        # Additional info
        info = {'score': score, 'lives': lives}
        truncated = False  # Modify as needed based on your environment's rules
        done = False # self._check_done()

        return pixels, reward, done, truncated, info

    def reset(self):
        """ Reset the state of the environment to an initial state.  """

        self.session.reset()
        self.session.run()
        initial_frame = self._get_frame()
        return initial_frame

    # def render(self, mode='human'):
    #     """ Render the environment.  """
    #     frame = self._get_frame()
    #     frame = frame.squeeze(0)# transform frame from (1,h,w) to (h,w)
    #     if self.render_mode == 'human':
    #         if not hasattr(self, 'screen') : self.screen = pygame.display.set_mode( (frame.shape[0], frame.shape[1]))
    #         pygame.surfarray.blit_array(self.screen, frame)
    #         pygame.display.flip()

    def render(self, mode=''):
        mode = self.render_mode if mode == '' else mode
        if mode == 'rgb_array':
            return np.transpose(self.pixel_history, (1, 2, 0))  # return last 3 frames as a single 'color-coded' image of motion
        elif mode == 'human':
            SCALE = 2
            if not hasattr(self, 'screen') or not pygame.get_init():
                pygame.init()
                self.screen = pygame.display.set_mode((self.WIDTH*SCALE, self.HEIGHT*SCALE))
                pygame.display.set_caption('MinJoustEnv Visualization')
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
                self.surface = pygame.Surface((self.pixel_history.shape[1], self.pixel_history.shape[2]))
            try:
                for event in pygame.event.get([pygame.QUIT]): pygame.quit(); return
                pygame.surfarray.blit_array(self.surface, np.transpose(self.pixel_history, (1, 2, 0)))
                transformed_surface = pygame.transform.flip(pygame.transform.rotate(self.surface, -90), True, False)
                scaled_surface = pygame.transform.scale(transformed_surface, (self.WIDTH*SCALE, self.HEIGHT*SCALE))
                self.screen.blit(scaled_surface, (0, 0))
                if not hasattr(self, 'last_text_surface') or (self.last_lives, self.last_score) != self.last_text_surface_value:
                    self.last_text_surface = self.font.render(f"Lives: {self.last_lives} Score: {self.last_score}", True, (255, 255, 255))
                    self.last_text_surface_value = (self.last_lives, self.last_score)
                self.screen.blit(self.last_text_surface, (10, 10))
                pygame.display.flip()
                self.clock.tick(60/self.FRAMES_PER_STEP)
            except (pygame.error, ZeroDivisionError) :
                pass

    def close(self):
        """ Perform any necessary cleanup.  """
        if self.render_mode == 'human':
            pygame.quit()
        self.session.__exit__(None, None, None)
        
    def _set_action(self, action):
        """ Convert and send the discrete action to the emulator.  """
        self.p1_input =self.actions[action]

    def _get_frame(self):
        """ Capture the current video frame from the emulator.  """
        framebuf = self.session.video.screenshot().data
        # framebuf = self.session.video._current._frame
        # unflatten to row,col,channel; # keep all rows&cols, but transform 'ABGR' to RGB...
        pixels1 = np.frombuffer(framebuf, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 4))[:,:,2::-1] 
        pixels2 = skimage.measure.block_reduce(pixels1, (self.DOWNSCALE,self.DOWNSCALE,3), np.mean) # downlsampled & grayscaled via mean;  shape now (h',w',1): 
        pixels3 = np.moveaxis(pixels2, -1, 0) #make channel first as per ML convention;   shape is now (1,h,w) 
        return pixels3 

    def _get_score(self):
        """ Retrieve the current score from the emulator's memory.  """
        memory = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        if memory is None:
            return 0
        score = int.from_bytes(memory[self.P1_SCORE_ADDR:self.P1_SCORE_ADDR+4], byteorder='little')# might need converting from hex encoded decimal 
        return score

    def _get_lives(self):
        """ Retrieve the current number of lives from the emulator's memory.  """
        memory = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        if memory is None:
            return 0
        lives = int.from_bytes(memory[self.P1_LIVES_ADDR], byteorder='little') # might need converting from hex encoded decimal 
        return lives

    def _check_done(self):
        """ Determine if the game has ended.  """
        lives = self._get_lives()
        return lives <= 0

    def _compute_reward(self, score, lives):
        """ Compute the reward based on the current state.  """
        # Example: Reward based solely on score
        return score


import time

# Example usage
if __name__ == "__main__":

    env = JoustEnv(render_mode='human')

    start_time = time.time()
    i = 0

    for episode in range(50_000_000):
        observation = env.reset()
        done, truncated, total_reward = False, False, 0
        while not (done or truncated):
            i += 1
            if i % (600) == 0: print(f"FPS: {i*env.FRAMES_PER_STEP // (time.time() - start_time)}")
            # action = env.action_space.sample()  # Replace with your agent's action
            # action = 3 if i%20==0 else 0 #[0,0,0,0,0,0,0,0,0,3][i%10]
            # action = 2 if i%40==0 else action#[0,0,0,0,0,0,0,0,0,3][i%10]
            action = [1,0,1,0][i%4]
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # env.render()
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = i / elapsed_time
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, FPS: {fps:.2f}")


    env.close()