""" A 'no-framework' RL environment for retro arcade game Joust """

import gymnasium as gym
import numpy as np
import pygame as pg
import libretro, skimage
import os, logging, time
from libretro import SessionBuilder, ExplicitPathDriver, JoypadState, InputDeviceFlag
from libretro.h import * 
from libretro.drivers import GeneratorInputDriver, UnformattedLogDriver, ModernGlVideoDriver
from itertools import repeat
import stable_baselines3 as sb3

class JoustEnv(gym.Env):
    """ Gym environment for the classic arcade game Joust using libretro.py.  """

    THROTTLE = False        # limit to 60 FTS (when in render_mode = 'human')?
    DOWNSCALE = 4           # Downscale the image by this factor (> 1 to speed up training)
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
    FULLY_LOADED_ADDR = 0x10207 #|U1  e8 = fully loaded
    PLAYING_ADDR = 0x00e290 #|U1  bit 7 :  0 = Attract, 1 = Playing

    NOOP, LEFT, RIGHT = JoypadState(), JoypadState(left=True), JoypadState(right=True)
    FLAP, FLAP_LEFT, FLAP_RIGHT = JoypadState(b=True), JoypadState(b=True, left=True), JoypadState(b=True, right=True)
    COIN, START = JoypadState(select=True), JoypadState(start=True)

    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self, render_mode=None):
        super(JoustEnv, self).__init__()

        # Initialize pygame for rendering
        pg.init()
        self.pg_clock = pg.time.Clock()

        # Define action/observation space
        self.action_inputs = [self.NOOP, self.FLAP, self.LEFT, self.RIGHT, self.FLAP_LEFT, self.FLAP_RIGHT]
        self.action_space = gym.spaces.Discrete(len(self.action_inputs))
        image_data_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.HEIGHT//self.DOWNSCALE, self.WIDTH//self.DOWNSCALE), 
            dtype=np.float32
        )
        input_history_space = gym.spaces.MultiBinary( (3, 3) ) # 3 inputs, 3 frames

        self.observation_space = gym.spaces.Dict({
            "image_data": image_data_space,
            "input_history": input_history_space
        })

        self.log_driver = UnformattedLogDriver()
        self.log_driver._logger.setLevel(logging.WARN)
        self.p1_input = JoypadState()

        self.render_mode = render_mode

        def input_callback(): 
            while True:
                yield self.p1_input

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

    def reset(self, seed=None):
        """ Reset the state of the environment to an initial state.  """
        self.session.reset()
        try: self._load_game(self.START_STATE_FILE)
        except: pass
        self.last_score = 0
        self.last_lives = self.START_LIVES 
        data_shape = (*self.observation_space['image_data'].shape, 3)
        self.pixel_history = np.zeros(data_shape, dtype=np.uint8)
        self.input_history = np.zeros((3,3), dtype=np.int8)
        self.mem = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)# hold a reference to core's memory 
        self.session.run()
        obs = self._get_observation()
        return (obs, {})


    def _get_observation(self):

        self.pixels = self._get_frame() # (use self persistant vars to avoid repeated allocations during tight-loop processing )
        square_shape = (self.WIDTH, self.WIDTH, 4) # _frame buffer comes back tall as it is wide, with empty bottom margin, and 4 channels
        margin = (self.WIDTH-self.HEIGHT) # margin is empty bottom

        # unflatten to row,col,channel; # keep all rows&cols, but transform 'ABGR' to RGB, and crop the margin:
        self.pixels0 = np.frombuffer(self.pixels, dtype=np.uint8).reshape(square_shape)[:-margin, :, 2::-1] 
        # just take one color channel # 1.5x faster than averaging. blue seems best.
        self.pixels1 = self.pixels0[:,:,1] # shape is now (h',w'): 
        # downsample: 
        self.pixels2 = skimage.measure.block_reduce(self.pixels1, (self.DOWNSCALE,self.DOWNSCALE), np.mean) # shape is now (h',w'): 

        # make monochrome: (values are: 0,255)        
        self.mono_pixels = (self.pixels2 > 26).astype(np.uint8) * 255 #  threshold 26 per common [.299,.587,.114] color->grayscale weights

        # keep a 3-frame history 
        self.pixel_history = np.roll(self.pixel_history, 1, axis=2)  # cycle 'channel' from [old, older, oldest] to [oldest, old, older]
        self.pixel_history[:,:,0] = self.mono_pixels # replace oldest with current

        # make a single frame with "trailers" to indicate motion
        self.viz_diff = self.pixel_history * np.array([1.0, 0.6, 0.3])[None,None,:] 
        self.viz_diff = np.max(self.viz_diff, axis=2).astype(np.uint8)

        # normalize it to 0,1
        self.image_data = self.viz_diff / 255

        # assemble inputs into 1-hot vector:
        one_hot_input = np.array([ 
            (self.p1_input.mask & self.FLAP.mask) > 0, 
            (self.p1_input.mask & self.LEFT.mask) > 0, 
            (self.p1_input.mask & self.RIGHT.mask) > 0, 
        ], dtype=np.uint8)
        self.input_history = np.roll(self.input_history, 1, axis=0)
        self.input_history[0] = one_hot_input

        # assemble observation:
        obs = { "image_data": self.image_data.astype(np.float32), "input_history": self.input_history }
        return obs



    def _get_reward(self):
        score = self._get_score()
        lives = self._get_lives()
        reward = lives - self.last_lives
        reward += (score - self.last_score) / 20_000 # there's a new life every 20k points so points are propotional 
        self.last_score = score 
        self.last_lives = lives 
        return reward


    def _get_truncated(self):   
        # playing = self.mem[self.PLAYING_ADDR] & 0b0100_0000 == 0
        # loaded = self.mem[self.FULLY_LOADED_ADDR] == 0xE8 
        # return loaded and not playing 
        return False

    def step(self, action=None):
        """ Execute one time step within the environment.  """
        if action: 
            self.p1_input = self.action_inputs[action] # self.p1_input is picked up by the emulator in input_callback
        
        if self.render_mode == "human": 
            self._process_input()# Override agent actions with user input 

        # step through _ frames (emulator will notice any new self.p1_input via the input_callback)
        for _ in range(self.FRAMES_PER_STEP): 
            self.session.run()

        obs = self._get_observation()
        rew = self._get_reward()
        trunc = self._get_truncated() 
        done = self._get_done()
        info = {'score': self.last_score, 'lives': self.last_lives}

        return obs, rew, done, trunc, info


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
                return self.pixel_history 
            case 'human':
                SCALE = 2
                if not hasattr(self, 'screen') : # first time render call
                    pg.display.set_mode((self.WIDTH*SCALE, self.HEIGHT*SCALE), pg.HWSURFACE) # hw surface ~2x faster ?
                    self.screen = pg.display.set_mode((self.WIDTH*SCALE, self.HEIGHT*SCALE))
                    pg.display.set_caption('JoustEnv Visualization')
                    self.font = pg.font.Font(pg.font.match_font('couriernew', bold=True), 14)
                    self.surface = pg.Surface(self.viz_diff.shape)#((self.pixel_history.shape[0], self.pixel_history.shape[1]))

                for event in pg.event.get([pg.QUIT]): pg.quit(); return
                pg.surfarray.blit_array(self.surface, self.viz_diff[...,None].repeat(3, axis=2))#self.pixel_history)
                transformed_surface = pg.transform.flip(pg.transform.rotate(self.surface, -90), True, False)
                scaled_surface = pg.transform.scale(transformed_surface, (self.WIDTH*SCALE, self.HEIGHT*SCALE))
                self.screen.blit(scaled_surface, (0, 0))
                self.stats_surface = self.font.render((
                    f"Lives:{self.last_lives} " 
                    f"Score:{self.last_score} " 
                    f"Coin:{self.mem[self.CREDITS_ADDR]} " 
                    f"Done:{self._get_done():01b} "
                    f"{self.input_history}" 
                ), True, (100, 100, 255))
                self.input_surface = self.font.render((
                    f"{self.p1_input.mask:016b} " 
                ), True, (100, 100, 255))
                self.screen.blit(self.stats_surface, (10, 10))
                self.screen.blit(self.input_surface, (10, self.HEIGHT*SCALE - 17))
                pg.display.flip()
                if self.THROTTLE: self.pg_clock.tick(60/self.FRAMES_PER_STEP) 


    def close(self):
        """ Perform any necessary cleanup.  """
        if self.render_mode == 'human':
            pg.quit()
        self.session.__exit__(None, None, None)
        

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
        return self.session.video._current._frame # this is more direct framebuffer access but yields square dimensions unconverted to RGB


    def _get_score(self):
        """ Retrieve the current score from the emulator's memory.  """
        # ______XX = self.mem[self.SCORE_LEAST_SIG_ADDR] 
        # ____XX__ = self.mem[self.SCORE_LESS_SIG_ADDR] 
        # __XX____ = self.mem[self.SCORE_MORE_SIG_ADDR] 
        # XX______ = self.mem[self.SCORE_MOST_SIG_ADDR] 
        # return = self._bcd_to_int(XX______)*1000000 + self._bcd_to_int(__XX____)*10000 + self._bcd_to_int(____XX__)*100 + self._bcd_to_int(______XX)
        # # Faster version:
        score_bytes = self.mem[self.SCORE_MOST_SIG_ADDR:self.SCORE_MOST_SIG_ADDR+5]
        return int(sum(self._bcd_to_int(b) * 10**(6-(2*i)) for i, b in enumerate(score_bytes)))


    def _get_lives(self):
        """ Retrieve the current number of lives from the emulator's memory.  """
        lives = self.mem[self.P1_LIVES_ADDR] 
        return lives

    def _get_done(self):
        return self.last_lives == 0


    def _print_memory_block(self):
        """ Print a block of memory for debugging purposes.  """
        self.mem = self.session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        for i in range(0, len(self.mem), 16):
            print(f"{i:04X}: {' '.join(f'{byte:02X}' for byte in self.mem[i:i+16])}")


    @staticmethod
    def _bcd_to_int(bcd_value):
        """ Convert BCD (Binary Coded Decimal) to a decimal int """
        # Old MAME roms often store numbers in memory as BCD
        # BCD amounts to "the hex formated number, read as decimal (after the 0x part)"
        # try: return int(hex(bcd_value)[2:])
        # except: return 0 # don't want this to fail when scrambled memory is ready during boot sequence
        return (bcd_value>>4)*10 + (bcd_value & 0x0F) # faster version


# Example usage
def example_loop():
    env = JoustEnv(render_mode='human')

    for epi_count in range(1_000):
        observation = env.reset()
        done, truncated, total_reward = False, False, 0
        epi_steps=0
        epi_start = time.time()
        while not (done or truncated):
            epi_steps += 1
            action = env.action_space.sample()  # Replace with trained agent's action
            # action = [1,0,1,0][i%4] # without interleaving actions, they don't repeat. 
            # action = None 
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if env.render_mode=='human': env.render()
        
        epi_secs = time.time() - epi_start
        aps = epi_steps / epi_secs
        fps = aps*env.FRAMES_PER_STEP 
        print(f"Epi {epi_count + 1}: Rew: {total_reward:.2f}, FPS: {fps:.0f}, APS: {aps:.0f}")

    env.close()


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, seed=0):
    def _init():
        env = JoustEnv(render_mode=None)  
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def sb3_ppo():
    num_envs = 16 
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    total_timesteps = 0
    
    if os.path.exists("ppo_joust.zip"):
        model = sb3.PPO.load("ppo_joust", env=env)
        print("Loaded existing model")
    else:
        model = sb3.PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard",
                        n_steps=2048 // num_envs)#, device='mps')  
        print("Created new model")

    for i in range(500):
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False, tb_log_name=f"PPO_run_{i}")
        total_timesteps += 1_000_000
        model.save("ppo_joust")
        print(f"Iteration {i+1}/500 completed. Total timesteps: {total_timesteps}")

    env.close()


if __name__ == "__main__":
    sb3_ppo()
    
