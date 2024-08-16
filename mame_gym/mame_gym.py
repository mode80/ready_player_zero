# %%
from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .mame_client import MAMEClient
import json
from textwrap import dedent
# from matplotlib import pyplot as plt

class JoustEnv(gym.Env):

    PLAYER = 1 # 1 or 2
    FPS = 60
    BOOT_SECONDS = 13 
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

        # Connect to a MAME instance launched with e.g "mame -autoboot_script mame_server.lua"
        self.mame = mame_client or MAMEClient()
        self.mame.connect()

        self._init_lua_globals()

        # Gather input info for this rom
        # self.inputs = self._get_rom_inputs() 
        # self.inputs = {
        #     ":IN0":{ "1 Player Start":32, "2 Players Start":16 },
        #     ":INP2": { "P2 Button 1":4, "P2 Left":1, "P2 Right":2 },
        #     ":INP1A":[],
        #     ":INP2A":[],
        #     ":INP1":{ "P1 Button 1":4, "P1 Right":2, "P1 Left":1 },
        #     ":IN2":{
        #         "Auto Up / Manual Down":1, "Coin 2":32,
        #         "Advance":2, "Coin 1":16, "High Score Reset":8,
        #         "Coin 3":4, "Tilt":64
        #     },
        #     ":IN1":[]
        # }

        # Define action space based on available inputs
        # self.action_space = spaces.Discrete(sum(len(port) for port in self.inputs.values()))
        # self.action_space = spaces.Discrete(6)  # Left, Right, Flap, Left+Flap, Right+Flap, No-op
        self.action_space = spaces.Discrete(4)  # Left, Right, Flap, No-op
        self.actions = [JoustEnv.LEFT, JoustEnv.RIGHT, JoustEnv.UP, None] # Joust specific

        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), dtype=np.uint8) 

        self.render_mode = None 
        self.reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))


    def step(self, action):
        
        action = self.actions[action]
        if action is not None:
            self._send_input(action)
        self._step_frame()
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_done()
        info = {}
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # actions to reset the mame emulation
        self._soft_reset()
        sleep(JoustEnv.BOOT_SECONDS) # wait for game to boot up 
        self.__init__() # required to reconnect the client etc after a MAME soft_reset
        self._ready_up() # actions to start the game
        while self._commands_are_processing(): pass # wait for game to start
        self._pause()
        self._throttled_off()
        
        # re-cache last score and lives for both players
        self.last_score = {1: 0, 2: 0}
        self.last_lives = {1: self._get_lives(), 2: 3}
        
        observation = self._get_observation()

        info = {}
        return observation, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_observation()
        elif self.render_mode == None:
            return None
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported.")

    def close(self):
        self.mame.close()

    def _queue_command(self, command):
        # adds a line (or many semi-colon delimited lines) of Lua code to a queue for execution over future frames
        self.mame.execute(f"commands=commands..'{command}'..';' ")

    def _ready_up(self):
        # Insert a coin
        self._send_input(JoustEnv.COIN1) # Joust specific
        # Press start button
        self._send_input(JoustEnv.START) # Joust specific

    def _wait_frames(n):
        # wait timespan of n frames
        sleep(n/JoustEnv.FPS) 

    def _send_input(self, port_field):
        # takes a (port,field) input tuple e.g. (":IN2","Coin 1") and simulates that user input 
        # these are rom specific and can be found with _get_inputs() 
        # this action is async in the sense that the inputs may be processed after this function returns
        port = port_field[0]
        field = port_field[1]
        self._queue_command((
            "ioports['{port}'].fields['{field}']:set_value(1); "
            "ioports['{port}'].fields['{field}']:set_value(0) "
        ))

    def _commands_are_processing(self):
        # returns true if there are commands in the queue that have not been processed
        ret = self.mame.execute("return #cmdQ > 0")
        return ret == b'true'

    def _pause(self):
        ret = self.mame.execute("emu.pause()")
        if ret != b'OK': raise RuntimeError(ret)

    def _unpause(self):
        ret = self.mame.execute("emu.unpause()")
        if ret != b'OK': raise RuntimeError(ret)

    def _throttled_on(self):
        ret = self.mame.execute("manager.machine.video.throttled = true")
        if ret != b'OK': raise RuntimeError(ret)

    def _throttled_off(self):
        ret = self.mame.execute("manager.machine.video.throttled = false")
        if ret != b'OK': raise RuntimeError(ret)

    def _step_frame(self):
        ret = self.mame.execute("emu.step()")
        if ret != b'OK': raise RuntimeError(ret)

    def _soft_reset(self):
        ret = self.mame.execute("return manager.machine:soft_reset()")
        if ret != b'OK': raise RuntimeError(ret)

    def _get_screen_size(self):
        result = self.mame.execute("return screen.width .. 'x' .. screen.height") # depends on _init_lua_globals for 'screen'
        width, height = map(int, result.decode().split('x'))
        return width, height

    def _get_frame_number(self):
        result = self.mame.execute("return screen:frame_number()") # depends on _init_lua_globals for 'screen'
        return int(result.decode())

    def _get_pixels(self):
        return self.mame.execute("return screen:pixels()") # depends on _init_lua_globals for 'screen'

    def _get_observation(self):
        pixels = self._get_pixels()
        # trim any "footer" data beyond pixel values of JoustEnv.HEIGHT * JoustEnv.WIDTH * 4
        pixels = pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4]
        # unflatten bytes into row,col,channel format; keep all rows and cols, but transform 'ABGR' to RGB  
        observation = np.frombuffer(pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4], 
                                    dtype=np.uint8).reshape((JoustEnv.HEIGHT, JoustEnv.WIDTH, 4))[:,:,2::-1]
        return observation

    def _read_byte(self, address):
        result = self.mame.execute(f"return mem:read_u8(0x{address:X})") # depends on _init_lua_globals for 'mem'
        return int(result.decode())

    def _read_word(self, address):
        result = self.mame.execute(f"return mem:read_u16(0x{address:X})") # depends on _init_lua_globals for 'mem'
        return int(result.decode())

    def _read_dword(self, address):
        result = self.mame.execute(f"return mem:read_u32(0x{address:X})") # depends on _init_lua_globals for 'mem'
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

    def _calculate_reward(self):
        # Get current score and lives
        current_score = self._get_score()
        current_lives = self._get_lives()
        
        # Calculate score,lives differences
        score_diff = current_score - self.last_score[JoustEnv.PLAYER]
        lives_diff = current_lives - self.last_lives[JoustEnv.PLAYER]
        
        # Update last score and lives
        self.last_score[JoustEnv.PLAYER] = current_score
        self.last_lives[JoustEnv.PLAYER] = current_lives
        
        # Reward for score increase
        reward = score_diff / JoustEnv.MAX_SCORE_DIFF  # Normalize score difference
        
        # Penalty for losing lives
        if lives_diff < 0:
            reward = -1.0  # Penalty for each life lost
        
        return reward

    def _check_done(self):
        current_lives = self._get_lives()
        return current_lives == 0  # Game is over when player has no lives left

    def _init_lua_globals(self):
        # set some persistant global variables in the MAME client session for later use
        self.mame.execute(( # this gets sent as semi-colon separated Lua code without linebreaks
            "screen = manager.machine.screens[':screen'] or manager.machine.screens:at(1) ; " # reference to the screen device
            "ioports = manager.machine.ioport.port ; " # reference to the ioports device
            "mem = manager.machine.devices[':maincpu'].spaces['program'] ; " # reference to the maincpu program space
            "commands = '' ; " #persistant global string takes semicolon-delimited Lua code for execution over successive frames 
            "errors = '' ; " #holds semicolon-delimited error messages from Lua code execution
            "last_result = nil ; " #holds the result of the last Lua code execution when executed over frames
            "cmdQ = {} ; "
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
                    "if #cmdQ>0 and screen:frame_number()%2==0 then "#If queue has commands and 2 frames have passed (some need this)
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
        return self.mame.execute("return last_result")

    def _get_lua_errors(self):
        # check for past errors in the Lua code execution
        result = self.mame.execute("return errors")
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


# Example usage
if __name__ == "__main__":
    env = JoustEnv()

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
