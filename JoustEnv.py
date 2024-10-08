""" A 'no-framework' RL environment for retro arcade game Joust """

from tkinter.filedialog import Open
import gymnasium as gym
import numpy as np
import pygame as pg
import libretro, skimage
import os, logging, time
from libretro import SessionBuilder, ExplicitPathDriver, JoypadState, InputDeviceFlag
from libretro.h import * 
from libretro.drivers import GeneratorInputDriver, UnformattedLogDriver, ModernGlVideoDriver
from itertools import repeat

class JoustEnv(gym.Env):
    """ Gym environment for the classic arcade game Joust using libretro.py.  """

    THROTTLE = False        # limit to 60 FTS (when in render_mode = 'human')?
    DOWNSCALE = 1           # Downscale the image by this factor (> 1 to speed up training)
    FRAMES_PER_STEP = 5     # 12 press-or-release actions (6 complete button presses) per second is comparable to human reflexes 
                            # Joust might react based on a count of input flags over the last [?] frames 
                            # but best way to handle this is probably to feed it a history of [?] previous inputs in the observation
                            # and let it figure out what sequences do what
    WIDTH, HEIGHT = 292, 240 # pixel dimensions of the screen for this rom
    START_LIVES = 5

    # ROM_PATH= '/Users/user/mame/roms/joust.zip'
    ROM_PATH= '/Users/user/Documents/RetroArch/fbneo/roms/arcade/joust.zip'
    CORE_PATH= '/Users/user/Library/Application Support/RetroArch/cores/fbneo_libretro.dylib' 
    # CORE_PATH= './ignore/FBNeo/src/burner/libretro/fbneo_libretro.dylib' # debug dylib. needs own save state
    START_STATE_FILE = './states/joust_start_1p.state' # use './states/joust_start_1p_debug.state' for debug dylib
    SAVE_PATH= '/Users/user/Documents/RetroArch/saves'
    SYSTEM_PATH = ASSETS_PATH = PLAYLIST_PATH = '/tmp' 

    P1_LIVES_ADDR = 0xE252#|U1      
    SCORE_MOST_SIG_ADDR = 0xE24C #|U1
    CREDITS_ADDR = 0xe2f2 #|U1
 
    NOOP, LEFT, RIGHT = JoypadState(), JoypadState(left=True), JoypadState(right=True)
    FLAP, FLAP_LEFT, FLAP_RIGHT, = JoypadState(b=True), JoypadState(b=True, left=True), JoypadState(b=True, right=True)
    COIN, START = JoypadState(select=True), JoypadState(start=True)

    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self, render_mode=None):
        super(JoustEnv, self).__init__()

        # Initialize Pygame
        pg.init()
        self.pg_clock = pg.time.Clock()

        # Define action space: Example for Joust 
        # Adjust the number of actions based on actual game controls
        # self.actions = [self.NOOP, self.FLAP]#, self.FLAP_LEFT, self.FLAP_RIGHT]
        self.actions = [self.NOOP, self.FLAP, self.LEFT, self.RIGHT, self.FLAP_LEFT, self.FLAP_RIGHT]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, self.HEIGHT//self.DOWNSCALE, self.WIDTH//self.DOWNSCALE), 
            dtype=np.uint8
        )

        self.log_driver = UnformattedLogDriver()
        self.log_driver._logger.setLevel(logging.WARN)
        self.p1_input = JoypadState()

        def input_callback(): 
            while True:
                yield self.p1_input

        self.last_score, self.last_lives = 0,0 

        # for rendering
        self.render_mode = render_mode
        self.pixel_history = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Initialize libretro session with configured drivers
        self.session = ( libretro
            .defaults(self.CORE_PATH)
            .with_content(self.ROM_PATH)
            .with_input(GeneratorInputDriver(
                input_callback,  
                max_users=1, #if max_useres != 1, we end up with double players, double inputs ¯_(ツ)_/¯
                device_capabilities=InputDeviceFlag.JOYPAD)) # not sure if this is needed
            .with_log(self.log_driver)
            .with_paths( ExplicitPathDriver(
                corepath = self.CORE_PATH, system = self.SYSTEM_PATH, 
                assets = self.ASSETS_PATH, save = self.SAVE_PATH, 
                playlist = self.PLAYLIST_PATH,
            ))
            .build()
        )

        self.session.__enter__()

    def step(self, action=None):
        """ Execute one time step within the environment.  """

        if action: 
            self._set_action(action)    
        
        if self.render_mode == "human": 
            self._process_input()# Override agent actions with user input 

        # step through the coming frames with input supplied by self.p1_input via input_callback
        for _ in range(self.FRAMES_PER_STEP): 
            self.session.run()

        # make 3-frame history
        pixels = self._get_frame() # resulting frame
        self.pixel_history = np.roll(self.pixel_history, 1, axis=0)  # cycle 'channel' from [old, older, oldest] to [oldest, old, older]
        self.pixel_history[0] = pixels # replace oldest with current observation

        # Extract state information
        score = self._get_score()
        lives = self._get_lives()

        # Define reward
        reward = lives - self.last_lives
        reward += (score - self.last_score) / 20_000 # there's a new life every 20k points so points are propotional 

        self.last_score = score 
        self.last_lives = lives 

        # Additional info
        info = {'score': score, 'lives': lives}
        truncated = False  # Modify as needed 
        done = (lives == 0) 

        return pixels, reward, done, truncated, info


    def reset(self):
        """ Reset the state of the environment to an initial state.  """
        self.session.reset()
        try: self._load_game(self.START_STATE_FILE)
        except: pass
        self.session.run()
        initial_frame = self._get_frame()
        self.mem = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)# hold a reference to core's memory 
        self.last_score = 0  
        self.last_lives = self.START_LIVES 
        return initial_frame


    def _load_game(self, game_file):
        """ Load the default 'ready-to-play' game and initialize the game state. """

        with open(game_file, 'rb') as f:
            buffer = f.read()

        state_loaded = self.session.core.unserialize(buffer)
        assert state_loaded


    def _save_game(self, game_file):
        """ Save the current game state to the specified path. """

        try:
            size = self.session.core.serialize_size()
            assert size > 0
            buffer = bytearray(size)
            did_serialize = self.session.core.serialize(buffer)
            assert did_serialize
            with open(game_file, 'wb') as f:
                f.write(buffer)
            print(f"Game saved successfully to {game_file}")

        except IOError as e:
            print(f"Error saving game: {e}")


    def render(self, mode=''):
        mode = self.render_mode if mode == '' else mode
        match mode:
            case 'rgb_array':
                return np.transpose(self.pixel_history, (1, 2, 0))  # return last 3 frames as a single 'color-coded' image of motion
            case 'human':
                SCALE = 2
                if not hasattr(self, 'screen') : # first time render call
                    pg.display.set_mode((self.WIDTH*SCALE, self.HEIGHT*SCALE), pg.HWSURFACE) # hw surface ~2x faster ?
                    self.screen = pg.display.set_mode((self.WIDTH*SCALE, self.HEIGHT*SCALE))
                    pg.display.set_caption('JoustEnv Visualization')
                    self.font = pg.font.Font(pg.font.match_font('couriernew', bold=True), 16)
                    self.tiny_font = pg.font.Font(pg.font.match_font('couriernew', bold=True), 14)
                    self.surface = pg.Surface((self.pixel_history.shape[1], self.pixel_history.shape[2]))

                for event in pg.event.get([pg.QUIT]): pg.quit(); return
                pg.surfarray.blit_array(self.surface, np.transpose(self.pixel_history, (1, 2, 0)))
                transformed_surface = pg.transform.flip(pg.transform.rotate(self.surface, -90), True, False)
                scaled_surface = pg.transform.scale(transformed_surface, (self.WIDTH*SCALE, self.HEIGHT*SCALE))
                self.screen.blit(scaled_surface, (0, 0))
                self.stats_surface = self.font.render(
                    f"Lives: {self.last_lives} Score: {self.last_score} Coin: {self.mem[self.CREDITS_ADDR]}",
                    True, (255, 255, 255)
                )
                self.input_surface = self.tiny_font.render(f"{self.p1_input}", True, (255, 255, 255))
                self.input_surface = self.tiny_font.render(f"{self.p1_input.mask:016b}", True, (255, 255, 255))
                self.screen.blit(self.stats_surface, (10, 10))
                self.screen.blit(self.input_surface, (10, self.HEIGHT*SCALE - 17))
                pg.display.flip()
                if self.THROTTLE: self.pg_clock.tick(60/self.FRAMES_PER_STEP) 


    def close(self):
        """ Perform any necessary cleanup.  """
        if self.render_mode == 'human':
            pg.quit()
        self.session.__exit__(None, None, None)
        
    def _set_action(self, action):
        """ Convert and set the discrete action for the emulator.  """
        self.p1_input = self.actions[action]

    def _process_input(self):
        """ Process Pygame events and update self.p1_input based on user input. """
        for event in pg.event.get():

            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.close()
                return

            if (event.type == pg.KEYDOWN and event.key == pg.K_F1):
                self._print_memory_block()

            if (event.type == pg.KEYDOWN and event.key == pg.K_F4):
                self._save_game(self.START_STATE_FILE)

            if (event.type == pg.KEYDOWN and event.key == pg.K_F5):
                self.mem[self.CREDITS_ADDR]=0x1

            elif event.type in [pg.KEYDOWN, pg.KEYUP] :
                keys = pg.key.get_pressed()
                key_mappings = {
                    'b': [pg.K_LCTRL, pg.K_SPACE, pg.K_z, pg.K_UP, pg.K_w, pg.K_LSHIFT],
                    'select': [pg.K_5, pg.K_TAB], 
                    'start': [pg.K_1, pg.K_RETURN],
                    'up': [pg.K_UP], 'down': [pg.K_DOWN],
                    'left': [pg.K_LEFT, pg.K_a], 'right': [pg.K_RIGHT, pg.K_d],
                    'a': [pg.K_a], 'x': [pg.K_x], 'y': [pg.K_y]
                }
                button_states = {button: any(keys[key] for key in key_list) for button, key_list in key_mappings.items()}
                new_input = JoypadState(**button_states)
                self.p1_input = new_input #if new_input != JoypadState() else self.NOOP


    def _get_frame(self):
        """ Capture the current video frame from the emulator.  """
        # framebuf = self.session.video.screenshot().data
        framebuf = self.session.video._current._frame # this is more direct framebuffer access but yields square dimensions unconverted to RGB
        square_shape = (self.WIDTH, self.WIDTH, 4) # _frame buffer is large enough to be as tall as it is wide with margins, and 4 channels
        margin = (self.WIDTH-self.HEIGHT)
        # unflatten to row,col,channel; # keep all rows&cols, but transform 'ABGR' to RGB, and crop off the top/bottom margins
        pixels1 = np.frombuffer(framebuf, dtype=np.uint8).reshape(square_shape)[:-margin, :, 2::-1] 
        # pixels2 = skimage.measure.block_reduce(pixels1, (self.DOWNSCALE,self.DOWNSCALE,3), np.mean) # downlsampled & grayscaled via mean;  shape now (h',w',1): 
        pixels2 = pixels1[:,:,1:2] # just take one color channel # 1.5x faster than mean
        pixels3 = np.moveaxis(pixels2, -1, 0) #make channel first as per ML convention;   shape is now (1,h,w) 
        return pixels3 

    def _get_score(self):
        """ Retrieve the current score from the emulator's memory.  """
        # ______XX = self.mem[self.SCORE_LEAST_SIG_ADDR] 
        # ____XX__ = self.mem[self.SCORE_LESS_SIG_ADDR] 
        # __XX____ = self.mem[self.SCORE_MORE_SIG_ADDR] 
        # XX______ = self.mem[self.SCORE_MOST_SIG_ADDR] 
        # return = self._bcd_to_int(XX______)*1000000 + self._bcd_to_int(__XX____)*10000 + self._bcd_to_int(____XX__)*100 + self._bcd_to_int(______XX)
        # # Faster version:
        score_bytes = self.mem[self.SCORE_MOST_SIG_ADDR:self.SCORE_MOST_SIG_ADDR+5]
        return sum(self._bcd_to_int(b) * 10**(6-(2*i)) for i, b in enumerate(score_bytes))

    def _get_lives(self):
        """ Retrieve the current number of lives from the emulator's memory.  """
        if self.mem is None: return 0
        lives = self.mem[self.P1_LIVES_ADDR] 
        return lives

    def _print_memory_block(self):
        """ Print a block of memory for debugging purposes.  """
        self.mem = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)# hold a reference to core's memory 
        if self.mem is None: return
        for i in range(0, len(self.mem), 16):
            print(f"{i:04X}: {' '.join(f'{byte:02X}' for byte in self.mem[i:i+16])}")

    def _check_done(self):
        """ Determine if the game has ended.  """
        lives = self._get_lives()
        return lives <= 0

    def _compute_reward(self, score, lives):
        """ Compute the reward based on the current state.  """
        # Example: Reward based solely on score
        return score

    @staticmethod
    def _bcd_to_int(bcd_value):
        """ Convert BCD (Binary Coded Decimal) to a decimal int """
        # Old MAME roms often store numbers in memory as BCD
        # BCD amounts to "the hex formated number, read as decimal (after the 0x part)"
        # try: return int(hex(bcd_value)[2:])
        # except: return 0 # don't want this to fail when scrambled memory is ready during boot sequence
        return (bcd_value >> 4) * 10 + (bcd_value & 0x0F) # faster version

# Example usage
if __name__ == "__main__":

    env = JoustEnv(render_mode='human')

    for epi_count in range(1_000_000):
        observation = env.reset()
        done, truncated, total_reward = False, False, 0
        epi_steps=0
        epi_start = time.time()
        while not (done or truncated):
            epi_steps += 1
            action = env.action_space.sample()  # Replace with trained agent's action
            # action = [1,0,1,0][i%4] # without interleaving actions, they don't repeat. need last_action as an input(?)
            # action = None 
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if env.render_mode=='human': env.render()
        
        epi_secs = time.time() - epi_start
        aps = epi_steps / epi_secs
        fps = aps*env.FRAMES_PER_STEP 
        print(f"Epi {epi_count + 1}: Rew: {total_reward:.2f}, FPS: {fps:.0f}, APS: {aps:.0f}")

    env.close()
