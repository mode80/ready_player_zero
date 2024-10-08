# %%
from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mame_client import MAMEClient
# from textwrap import dedent
# from matplotlib import pyplot as plt

class JoustEnv(gym.Env):

    PLAYER = 1 # 1 or 2
    BOOT_SECONDS = 10 # Number of seconds after MAME reboot before client can connect 
    BOOT_SECS_UNTHROTTLED = 2 #  "   "   " when not throttled
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. TODO: End of stage bonus?
    READY_UP_FRAMES = 150 # How many frames after "pressing start" before player can move

    WIDTH=292 # screen pixel dimensions  
    HEIGHT=240

    # Memory addresses for Joust
    # to find similar for other roms see https://docs.mamedev.org/debugger/cheats.html 
    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
    P2_LIVES_ADDR = 0xA05C
    P2_SCORE_ADDR = 0xA058

    # Input Actions are a list of Input Items, each having (port, field, value, n_frames_from_now)
    # where 'port' and 'field' are rom specific, found with _get_rom_inputs(), 
    # 'value' is 1 or 0 (for buttons) to indicate "press down" and "release" respectively, 
    # and 'n_frames_from_now' is how many frames to wait before applying the input (0 for right now)
    # a List of these can simulate complex input like "press button, hold joystick up and to the left, release".
    # the full Input Action sequence is sent & executed in the MAME Lua environment to avoid IO timing variance 

    COIN1           = [(':IN2' , 'Coin 1',         1, 0),   (':IN2', 'Coin 1',         0, 2)] # press button, release in 2 frames
    START           = [(':IN0' , '1 Player Start', 1, 0),   (':IN0', '1 Player Start', 0, 2)]
    START_2P        = [(':IN0' , '2 Player Start', 1, 0),   (':IN0', '2 Player Start', 0, 1)]

    P1_FLAP         = [(':INP1', 'P1 Button 1',    1, 0),   (':INP1','P1 Button 1',    0, 1)] # press flap, release next frame
    P1_STICK_LEFT   = [(':INP1', 'P1 Left',        1, 0),   (':INP1', 'P1 Right',      0, 0)] # when stick is left, stick-right is forced off
    P1_STICK_RIGHT  = [(':INP1', 'P1 Right',       1, 0),   (':INP1', 'P1 Left',       0, 0)] # and vice versa
    P1_STICK_CENTER = [(':INP1', 'P1 Left',        0, 0),   (':INP1','P1 Right',       0, 0)] # center stick releases both left and right

    P2_FLAP         = [(':INP2', 'P2 Button 1',    1, 0),   (':INP2','P2 Button 1',    0, 1)] 
    P2_STICK_LEFT   = [(':INP2', 'P2 Left',        1, 0),   (':INP2', 'P2 Right',      0, 0)] 
    P2_STICK_RIGHT  = [(':INP2', 'P2 Right',       1, 0),   (':INP2', 'P2 Left',       0, 0)] 
    P2_STICK_CENTER = [(':INP2', 'P2 Left',        0, 0),   (':INP2','P2 Right',       0, 0)]

    P1_FLAP_LEFT   = P1_FLAP + P1_STICK_LEFT  # press flap, hold stick left
    P1_FLAP_RIGHT  = P1_FLAP + P1_STICK_RIGHT # press flap, hold stick right
    P1_FLAP_CENTER = P1_FLAP + P1_STICK_CENTER # press flap, hold stick center

    P2_FLAP_LEFT   = P2_FLAP + P2_STICK_LEFT  
    P2_FLAP_RIGHT  = P2_FLAP + P2_STICK_RIGHT
    P2_FLAP_CENTER = P2_FLAP + P2_STICK_CENTER

    P1_NOOP            = P1_STICK_CENTER 
    P2_NOOP            = P2_STICK_CENTER

    if PLAYER == 1:
        NOOP, FLAP, LEFT, RIGHT, FLAP_LEFT, FLAP_RIGHT, FLAP_CENTER = P1_NOOP, P1_FLAP, P1_STICK_LEFT, P1_STICK_RIGHT, P1_FLAP_LEFT, P1_FLAP_RIGHT, P1_FLAP_CENTER
    else:
        NOOP, FLAP, LEFT, RIGHT, FLAP_LEFT, FLAP_RIGHT, FLAP_CENTER = P2_NOOP, P2_FLAP, P2_STICK_LEFT, P2_STICK_RIGHT, P2_FLAP_LEFT, P2_FLAP_RIGHT, P2_FLAP_CENTER


    def __init__(self, mame_client=None):
        super().__init__()
        self.client = mame_client or MAMEClient()

        # Define action space based on available inputs
        self.action_space = spaces.Discrete(7)  
        self.actions = [JoustEnv.NOOP, JoustEnv.FLAP, JoustEnv.LEFT, JoustEnv.RIGHT, JoustEnv.FLAP_LEFT, JoustEnv.FLAP_RIGHT, JoustEnv.FLAP_CENTER]

        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), dtype=np.uint8) 

        self.render_mode = None 
        self.reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))
        self.is_paused = False

        # Connect to a MAME instance already launched with e.g "mame -autoboot_script mame_server.lua"
        self.client.connect() 
        self._init_lua()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._throttled_off()
        self._soft_reset()
        sleep(JoustEnv.BOOT_SECS_UNTHROTTLED) # wait for reboot TODO: try connecting repeately instead of sleeping
        self.client.connect() # reconnect after reboot
        self._throttled_on()
        self._ready_up() # actions to start the game
        self._pause()

        # re-cache last score and lives 
        self.last_score = self._get_score() 
        self.last_lives = self._get_lives()
               
        observation = self._get_image()

        info = {}
        return observation, info

    def step(self, action):
        command = self.actions[action]
        print(command) # TODO remove debug print
        if self.is_paused: self._unpause() 
        if command: self._send_input(command) # this has the behaviour of queuing but not executing until 1 frame later 

        self._pause()
        observation = self._get_image()
        lives = self._get_lives()
        score = self._get_score()
        reward = self._calculate_reward(score,lives)
        done = self._check_done(score,lives)
        info = {'lives': lives, 'score': score}
        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_image()
        elif self.render_mode == None:
            return None
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported.")

    def close(self):
        self.client.close()

    def _is_ready(self):
        # Asks the MAME instance if it is paused. Can also mean it's booting up so really represents "is ready"
        ret = self.client.execute("return machine.paused()")
        return self.is_paused 

    def _ready_up(self):
        self._send_input(JoustEnv.COIN1) # insert a coin # Joust specific
        self._send_input(JoustEnv.START) # press Start # Joust specific
        self._wait_n_frames(JoustEnv.READY_UP_FRAMES)  
        self.client.execute("print('done ready up')")
        # while not self._is_ready: pass  

    def _send_input(self, input_action):
        # takes an input action data structure and simulates the corrsponding user input
        # e.g. [(':IN2', 'Coin 1', 1, 0), (':IN2', 'Coin 1', 0, 2)] # press insert coin button, release in 2 frames
        lua = ""
        for ia in input_action:
            if ia[3] == 0: # "this frame actions"
                lua += f"ioports['{ia[0]}'].fields['{ia[1]}']:set_value({ia[2]}) ; "
            else: # future frame actions
                lua += f"inputs=inputs..\"{str(ia)}\"; " # set the lua global for later pickup in frame loop running on Lua side
        if lua:
            was_paused = self.is_paused 
            if was_paused: self._unpause()
            self.client.execute(lua)   
            if was_paused: self._pause()

    def _commands_are_processing(self):
        # returns true if there are commands in the queue that have not been processed
        ret = self.client.execute("return #cmdQ > 0")
        return ret == b'true'

    def _pause(self):
        ret = self.client.execute("emu.pause()")
        if ret != b'OK': raise RuntimeError(ret)
        self.is_paused = True

    def _unpause(self):
        ret = self.client.execute("emu.unpause()")
        if ret != b'OK': raise RuntimeError(ret)
        self.is_paused = False 

    def _wait_n_frames(self, n):
        # wait timespan of n frame numbers
        was_paused = self.is_paused
        if was_paused: self._unpause() # frame numbers don't increment when paused
        init_frame_num = self._get_frame_number()
        end_frame_num = init_frame_num + n - 1 # -1 because initial call to _get_frame_number() will increment the frame number
        while self._get_frame_number() < end_frame_num: pass
        if was_paused: self._pause()

    def _throttled_on(self):
        ret = self.client.execute("manager.machine.video.throttled = true")
        if ret != b'OK': raise RuntimeError(ret)

    def _throttled_off(self):
        ret = self.client.execute("manager.machine.video.throttled = false")
        if ret != b'OK': raise RuntimeError(ret)

    def _wait(self, secs):
        ret = self.client.execute(f"return emu.wait({secs})")
        return (ret == b'true')

    def _soft_reset(self):
        ret = self.client.execute("return manager.machine:soft_reset()")
        if ret != b'OK': raise RuntimeError(ret)
        self.is_paused=False

    def _get_screen_size(self):
        result = self.client.execute("return screen.width .. 'x' .. screen.height") # depends on _init_lua for 'screen'
        width, height = map(int, result.decode().split('x'))
        return width, height

    def _get_frame_number(self):
        result = self.client.execute("return screen:frame_number()") # depends on _init_lua for 'screen'
        return int(result.decode())

    def _get_pixels(self):
        return self.client.execute("return screen:pixels()") # depends on _init_lua for 'screen'

    def _get_image(self):
        pixels = self._get_pixels()
        # trim any "footer" data beyond pixel values of JoustEnv.HEIGHT * JoustEnv.WIDTH * 4
        pixels = pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4]
        # unflatten bytes into row,col,channel format; keep all rows and cols, but transform 'ABGR' to RGB  
        observation = np.frombuffer(pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4], 
                                    dtype=np.uint8).reshape((JoustEnv.HEIGHT, JoustEnv.WIDTH, 4))[:,:,2::-1]
        return observation

    def _read_byte(self, address):
        result = self.client.execute(f"return mem:read_u8(0x{address:X})") # depends on _init_lua for 'mem'
        return int(result.decode())

    def _read_word(self, address):
        result = self.client.execute(f"return mem:read_u16(0x{address:X})") # depends on _init_lua for 'mem'
        return int(result.decode())

    def _read_dword(self, address):
        result = self.client.execute(f"return mem:read_u32(0x{address:X})") # depends on _init_lua for 'mem'
        return int(result.decode())

    def _get_lives(self):
        if JoustEnv.PLAYER==2:
            return self._read_byte(self.P2_LIVES_ADDR)
        else: 
            return self._read_byte(self.P1_LIVES_ADDR)

    def _get_score(self):
        if JoustEnv.PLAYER==2:
            return JoustEnv._bcd_to_int( self._read_dword(self.P2_SCORE_ADDR) )
        else: 
            return JoustEnv._bcd_to_int( self._read_dword(self.P1_SCORE_ADDR) )

    def _calculate_reward(self, score=None, lives=None):
        # Get current score and lives
        current_score = score #or self._get_score() # shouldn't run _get_score but does?
        current_lives = lives #or self._get_lives()
        
        # Calculate score,lives differences
        score_diff = current_score - self.last_score
        lives_diff = current_lives - self.last_lives
        
        # Update last score and lives
        self.last_score = current_score
        self.last_lives = current_lives
        
        # Reward for score increase
        reward = score_diff / JoustEnv.MAX_SCORE_DIFF  # Normalize score difference
        
        # Penalty for losing lives
        if lives_diff < 0:
            reward = -1.0  # Penalty for each life lost
        
        return reward

    def _check_done(self, score=None, lives=None):
        current_lives = lives #or self._get_lives()
        current_score = score #or self._get_score()
        return current_lives == 0 and current_score > 0 # score check here prevents false trigger at game start

    def _init_lua(self):
        # set some persistant global variables in the MAME client session for later use
        self.client.execute(( # this gets sent as semi-colon separated Lua code without linebreaks
            "screen = manager.machine.screens:at(1) ; " # reference to the screen device
            "ioports = manager.machine.ioport.ports ; " # reference to the ioports device
            "mem = manager.machine.devices[':maincpu'].spaces['program'] ; " # reference to the maincpu program space
            "printt = function(t) for k,v in pairs(t) do print(k,v) end end ; " # helper function to print tables
            # enables setting the 'inputs' global string with comma-delimited input data for processing over successive frames
            # a sample 'inputs' string:  [(':IN2' ,'Coin 1', 1, 0), (':IN2', 'Coin 1', 0, 2)] 
            # each parens group contains: (ioport, iofield, value, n_frame_from_now)
            "last_frame = 0 ; "
            "inputs = '' ; " 
            "input_list = {} ; "
            "ponce = 0 ; "
            "if inputs_sub then inputs_sub:unsubscribe() end ; " #remove any existing frame notifier for inputs 
        )) # there's a limit on how much text can execute in one call, so this is split into multiple calls
        self.client.execute(( # this gets sent as semi-colon separated Lua code without linebreaks
            "inputs_sub = emu.add_machine_frame_notifier( "
                "function() "
                    "this_frame = screen:frame_number() ; "
                    "is_paused = (this_frame == last_frame) ; "
                    "if is_paused then ponce = ponce + 1 else ponce = 0 end ; "
                    "print(this_frame..':'..ponce) ; "
                    "if (string.len(inputs) > 0) then " #-- Check if the inputs string is non-empty # TODO  this could/should me moved to execute same-frame for 0-delay inputs
                        #-- Pull contents of each () group into a table
                        "for input_action in string.gmatch(inputs, '%s*%((.-)%)') do "
                            #-- Split the 4 comma-delimited string and number values into 4 named variables
                            "a, b, c, d = string.match(input_action, \"%s*'([^']*)'%s*,%s*'([^']*)'%s*,%s*(%d+)%s*,%s*(%d+)%s*\") ; "
                            #-- Create a table for the input action
                            "input_map = {ioport=a, iofield=b, value=tonumber(c), on_frame=this_frame+tonumber(d)-1} ; " # -1 because it was scheduled last frame
                            #-- Add this to the input list 
                            "table.insert(input_list, input_map) ; "
                        "end "
                        "inputs='' ; " # reset inputs 
                    "end "
                    "if #input_list>0 and not is_paused then " #--if there are inputs to process
                        "for i, input in ipairs(input_list) do " #--iterate over the input list
                            "if input.on_frame <= this_frame then " #--if the input is scheduled for now (or previously)
                                "ioports[input.ioport].fields[input.iofield]:set_value(input.value) ; "  #--action the input
                                "print('', input.ioport, input.iofield, input.value) ; "  #--log
                                "table.remove(input_list, i) ; " #--this input is no longer pending
                            "end "
                        "end "
                    "end "
                "last_frame = this_frame ; "#-- notifier runs even when game is paused, so ensure new frame number above
                "end "
            ") "
        ))

    def _get_lua_last_result(self):
        # return result of the last frame-queued command on the Lua side 
        return self.client.execute("return last_result")

    def _get_lua_errors(self):
        # check for past errors in the Lua code execution
        result = self.client.execute("return errors")
        if result != b'': raise Exception(result.decode())

    def _bcd_to_int(bcd_value):
        # Convert BCD (Binary Coded Decimal) to a decimal int
        # Old MAME roms often store numbers in memory as BCD
        # BCD amounts to "the hex formated number, read as decimal (after the 0x part)"
        return int(hex(bcd_value)[2:])

    # def _get_rom_inputs(self):
    #     # utility fn to get all available "port" and "field" input codes for arbitraty MAME roms 
    #     lua_script = """
    #         function serialize(obj)
    #             local items = {}
    #             for k, v in pairs(obj) do
    #                 if type(v) == "table" then
    #                     items[#items+1] = string.format("%q:{%s}", k, serialize(v))
    #                 else
    #                     items[#items+1] = string.format("%q:%q", k, tostring(v))
    #                 end
    #             end
    #             return "{" .. table.concat(items, ",") .. "}"
    #         end
    #         function get_inputs()
    #             local inputs = {}
    #             for port_name, port in pairs(manager.machine.ioport.ports) do
    #                 inputs[port_name] = {}
    #                 for field_name, field in pairs(port.fields) do
    #                     inputs[port_name][field_name] = field.mask
    #                 end
    #             end
    #             return serialize(inputs)
    #         end
    #         return get_inputs()
    #     """
    #     return self.client.execute(lua_script)

# def sample_PPO():
#     from stable_baselines3 import PPO

#     env = JoustEnv()
#     model = PPO("CnnPolicy", env, verbose=1)
#     model.learn(total_timesteps=100000)

#     # Test the trained model
#     obs, _ = env.reset()
#     for _ in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         if done or truncated:
#             obs, _ = env.reset()

#     env.close()


# Example usage
if __name__ == "__main__":
    env = JoustEnv()
    env.reset()

    for _ in range(1000): 
        action = env.action_space.sample()  # Random action
        # action = [2,2,0,0,0,3,3,3,3,3,0,0,0,2,2,2,2,2,0,0,0,3,3,3,3,3,0,0,0,2,2,2][_ % 32] # 5-steps to reverse direction in place. 5 steps after the 1st animates
        # action = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][_ % 25] # ~25 steps to flap once and land. 2nd frame after the flap animates. 
        observation, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            observation, info = env.reset()

    env.close()

