# %%
from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mame_client import MAMEClient
# import json
# from textwrap import dedent
# from matplotlib import pyplot as plt

class JoustEnv(gym.Env):

    PLAYER = 1 # 1 or 2
    FPS = 60
    BOOT_SECONDS = 11 # Number of seconds after reboot before game can receive input 
    BOOT_SECS_UNTHROTTLED = 2 #  "   "   " when not throttled
    INPUT_DELAY = 1/FPS
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. TODO: End of stage bonus?

    WIDTH=292
    HEIGHT=240

    # Memory addresses for Joust
    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
    P2_LIVES_ADDR = 0xA05C
    P2_SCORE_ADDR = 0xA058

    COIN1 = (":IN2", "Coin 1")
    P1_START = (":IN0", "1 Player Start")
    P1_LEFT = (":INP1", "P1 Left")
    P1_RIGHT = (":INP1", "P1 Right")
    P1_UP = (":INP1", "P1 Button 1")
    P2_START = (":IN0", "2 Players Start")
    P2_LEFT = (":INP2", "P2 Left")
    P2_RIGHT = (":INP2", "P2 Right")
    P2_UP = (":INP2", "P2 Button 1")

    if PLAYER == 1:
        START, LEFT, RIGHT, UP, = P1_START, P1_LEFT, P1_RIGHT, P1_UP
    else:
        START, LEFT, RIGHT, UP, = P2_START, P2_LEFT, P2_RIGHT, P2_UP


    def __init__(self, mame_client=None):
        super().__init__()
        self.client = mame_client or MAMEClient()

        # Define action space based on available inputs
        # self.action_space = spaces.Discrete(sum(len(port) for port in self.inputs.values()))
        # self.action_space = spaces.Discrete(6)  # Left, Right, Flap, Left+Flap, Right+Flap, No-op
        self.action_space = spaces.Discrete(4)  # Left, Right, Flap, No-op
        self.actions = [None, JoustEnv.UP, JoustEnv.LEFT, JoustEnv.RIGHT] # Joust specific

        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), dtype=np.uint8) 

        self.render_mode = None 
        self.reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))

        # Connect to a MAME instance launched with e.g "mame -autoboot_script mame_server.lua"
        self.client.connect() 
        self._init_lua_globals()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._throttled_off()
        self._reboot()
        sleep(JoustEnv.BOOT_SECS_UNTHROTTLED) # wait for reboot -- can't be based on state because connection for reading state is lost after reboot  
        self.client.connect() # reconnect after reboot
        self._init_lua_globals()
        # self._throttled_on()
        self._ready_up() # actions to start the game
        # self._pause()

        # re-cache last score and lives 
        self.last_score = self._get_score() 
        self.last_lives = self._get_lives()
               
        observation = self._get_observation()

        info = {}
        return observation, info

    def step(self, action):
        
        command = self.actions[action]
        print(command) # TODO remove debug print
        if command is not None:
            self._queue_input(command)
        for i in range(2):  # watching Joust indicates actions are not observable until the _th frame after input
            self._unpause() # if we don't unpause input doesn't process (TODO: uncomfortably unpredictable time unpaused due to IO) 
            self._step_frame() 
            # self._queue_command("emu.unpause();emu.step();")
        observation = self._get_observation()
        lives = self._get_lives()
        score = self._get_score()
        reward = self._calculate_reward(score,lives)
        done = self._check_done(score,lives)
        info = {}
        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_observation()
        elif self.render_mode == None:
            return None
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported.")

    def close(self):
        self.client.close()

    def _queue_command(self, command):
        # adds a line (or many semi-colon delimited lines) of Lua code to a queue for execution over future frames
        # the mechanism here is to set a global variable in Lua, which gets split up and executed frame by frame on the Lua side 
        # via the Lua code found in _init_lua_globabls
        # NOTE command should be valid Lua code, and should not contain line-feeds or double-quotes 
        self.client.execute(f"commands=commands..\"{command}\" ")

    def _ready_up(self):
        sleep(.1) # :/ 
        self._queue_input(JoustEnv.COIN1) # insert a coin # Joust specific
        sleep(.1) # why need this ?? :/
        self._queue_input(JoustEnv.START) # press Start # Joust specific
        # while self._get_lives() == 0 or self._get_score()!=0 : 
        #     sleep(JoustEnv.INPUT_DELAY)# wait for play to start
        # self._wait_n_frames(2)
        # while self._commands_are_processing(): pass # wait for play to start

    def _send_input(self, port_field):
        # takes a (port,field) input tuple e.g. (":IN2","Coin 1") and simulates that user input
        # these are rom specific and can be found with _get_inputs()
        # this action is async in the sense that the inputs may be processed after this function returns
        port = port_field[0]
        field = port_field[1]
        self.client.execute(f"ioports['{port}'].fields['{field}']:set_value(1); ")
        sleep(JoustEnv.INPUT_DELAY) # needs some delay betweeen input commands :/ 
        self.client.execute(f"ioports['{port}'].fields['{field}']:set_value(0); ")
        sleep(JoustEnv.INPUT_DELAY)

    def _queue_input(self, port_field):
        # takes a (port,field) input tuple e.g. (":IN2","Coin 1") and simulates that user input 
        # these are rom specific and can be found with _get_inputs() 
        # this action is async in the sense that the inputs may be processed after this function returns
        port = port_field[0]
        field = port_field[1]
        command = ((
            f"ioports['{port}'].fields['{field}']:set_value(1); "
            f"ioports['{port}'].fields['{field}']:set_value(0) "
        ))
        self._queue_command(command)

    def _commands_are_processing(self):
        # returns true if there are commands in the queue that have not been processed
        ret = self.client.execute("return #cmdQ > 0")
        return ret == b'true'

    def _pause(self):
        self.is_paused = True
        ret = self.client.execute("emu.pause()")
        if ret != b'OK': raise RuntimeError(ret)

    def _unpause(self):
        self.is_paused = False 
        ret = self.client.execute("emu.unpause()")
        if ret != b'OK': raise RuntimeError(ret)

    def _wait_n_frames(self, n):
        # wait timespan of n frames
        init_frame_num = self._get_frame_number()
        end_frame_num = init_frame_num + n
        while self._get_frame_number() < end_frame_num:
            pass

    def _throttled_on(self):
        ret = self.client.execute("manager.machine.video.throttled = true")
        if ret != b'OK': raise RuntimeError(ret)

    def _throttled_off(self):
        ret = self.client.execute("manager.machine.video.throttled = false")
        if ret != b'OK': raise RuntimeError(ret)

    def _step_frame(self):
        ret = self.client.execute("emu.step()")
        if ret != b'OK': raise RuntimeError(ret)

    def _reboot(self):
        ret = self.client.execute("return manager.machine:soft_reset()")
        if ret != b'OK': raise RuntimeError(ret)

    def _get_screen_size(self):
        result = self.client.execute("return screen.width .. 'x' .. screen.height") # depends on _init_lua_globals for 'screen'
        width, height = map(int, result.decode().split('x'))
        return width, height

    def _get_frame_number(self):
        result = self.client.execute("return screen:frame_number()") # depends on _init_lua_globals for 'screen'
        return int(result.decode())

    def _get_pixels(self):
        return self.client.execute("return screen:pixels()") # depends on _init_lua_globals for 'screen'

    def _get_observation(self):
        pixels = self._get_pixels()
        # trim any "footer" data beyond pixel values of JoustEnv.HEIGHT * JoustEnv.WIDTH * 4
        pixels = pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4]
        # unflatten bytes into row,col,channel format; keep all rows and cols, but transform 'ABGR' to RGB  
        observation = np.frombuffer(pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4], 
                                    dtype=np.uint8).reshape((JoustEnv.HEIGHT, JoustEnv.WIDTH, 4))[:,:,2::-1]
        return observation

    def _read_byte(self, address):
        result = self.client.execute(f"return mem:read_u8(0x{address:X})") # depends on _init_lua_globals for 'mem'
        return int(result.decode())

    def _read_word(self, address):
        result = self.client.execute(f"return mem:read_u16(0x{address:X})") # depends on _init_lua_globals for 'mem'
        return int(result.decode())

    def _read_dword(self, address):
        result = self.client.execute(f"return mem:read_u32(0x{address:X})") # depends on _init_lua_globals for 'mem'
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

    def _init_lua_globals(self):
        # set some persistant global variables in the MAME client session for later use
        self.client.execute(( # this gets sent as semi-colon separated Lua code without linebreaks
            "screen = manager.machine.screens[':screen'] or manager.machine.screens:at(1) ; " # reference to the screen device
            "ioports = manager.machine.ioport.ports ; " # reference to the ioports device
            "mem = manager.machine.devices[':maincpu'].spaces['program'] ; " # reference to the maincpu program space
            "commands = '' ; " #persistant global string takes semicolon-delimited Lua code for execution over successive frames 
            "errors = '' ; " #holds semicolon-delimited error messages from Lua code execution
            "last_result = nil ; " #holds the result of the last Lua code execution when executed over frames
            "cmdQ = {} ; "
            "last_frame = 0 ; "
            #below enables issuing a sequence of Lua instructions over successive future frames by setting the 'commands' global
            "process_commands_sub = emu.add_machine_frame_notifier( "
                "function() "
                    "if (string.len(commands) > 0) then "#Check global commands string for new commands
                        "for cmd in string.gmatch(commands, '[^;]+') do " #Split command string by ';' and iterate over each part
                            "if string.len(cmd) > 0 then  "#If the command is not empty
                                "cmdFn = load(cmd) ; "#Create a function for the command 
                                "table.insert(cmdQ, cmdFn); "#Add it the queue for execution
                            "end "
                        "end "
                        "commands = '' ; " #Clear the commands string for the next frame
                    "end "
                    "frame_num = screen:frame_number() ; " #Get the current frame number
                    "if #cmdQ>0 and (frame_num-last_frame)>=2 then "#If queue has commands and some frames have passed (frame spacing no less than this works) 
                        "last_frame = frame_num ; "
                        "success, last_result = pcall(cmdQ[1]) ; "#Execute the next command function in the queue with error checking
                        "if not success then "
                            "errors = errors..last_result..';'  ; "#Add error message to errors string
                            "cmdQ={} ; " #Clear the command queue after error
                        "end "
                        "table.remove(cmdQ,1) ; " #remove the function from the table after executing it
                    "end "
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
    #     lua_script = dedent("""
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
    #     """)
    #     result = self.mame.execute(lua_script)
    #     # Parse the JSON string into a Python dictionary and return
    #     input_dict = json.loads(result.decode())
    #     return input_dict

    # def _action_to_input(self, action):
    #     count = 0
    #     for port_name, fields in self.inputs.items():
    #         for field_name in fields:
    #             if count == action:
    #                 return port_name, field_name
    #             count += 1
    #     raise ValueError(f"Invalid action: {action}")

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

    observation, info = env.reset()

    for _ in range(1000):
        # action = env.action_space.sample()  # Random action
        action = [0,0,0,0,0,2,0,0,0,0,0,3][_ % 12]  
        observation, reward, done, truncated, info = env.step(action)
        sleep(.4)
        
        if done or truncated:
            observation, info = env.reset()

    env.close()

