from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .mame_client import MAMEClient
import json
from textwrap import dedent

class JoustEnv(gym.Env):

    # Memory addresses for Joust
    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
    P2_LIVES_ADDR = 0xA05C
    P2_SCORE_ADDR = 0xA058

    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. TODO: End of stage bonus?

    WIDTH=292
    HEIGHT=240

    COIN1 = (":IN2", "Coin 1")
    COIN2 = (":IN2", "Coin 2")
    P1_START = (":IN0", "1 Player Start")
    P2_START = (":IN0", "2 Players Start")
    P1_LEFT = (":INP1", "P1 Left")
    P1_RIGHT = (":INP1", "P1 Right")
    P1_UP = (":INP1", "P1 Button 1")

    FPS = 60
    BOOT_SECONDS = 8


    def __init__(self, mame_client=None):
        super().__init__()

        # Connect to a MAME instance launched with e.g "mame -autoboot_script mame_server.lua"
        self.mame = MAMEClient()
        self.mame.connect()

        # Gather input info for this rom
        # self.inputs = self._get_rom_inputs() 
        self.inputs = {
            ":IN0":{ "1 Player Start":32, "2 Players Start":16 },
            ":INP2": { "P2 Button 1":4, "P2 Left":1, "P2 Right":2 },
            ":INP1A":[],
            ":INP2A":[],
            ":INP1":{ "P1 Button 1":4, "P1 Right":2, "P1 Left":1 },
            ":IN2":{
                "Auto Up / Manual Down":1, "Coin 2":32,
                "Advance":2, "Coin 1":16, "High Score Reset":8,
                "Coin 3":4, "Tilt":64
            },
            ":IN1":[]
        }

        # Define action space based on available inputs
        # self.action_space = spaces.Discrete(sum(len(port) for port in self.inputs.values()))
        # self.action_space = spaces.Discrete(6)  # Left, Right, Flap, Left+Flap, Right+Flap, No-op
        self.action_space = spaces.Discrete(4)  # Left, Right, Flap, No-op
        self.actions = [JoustEnv.P1_LEFT, JoustEnv.P1_RIGHT, JoustEnv.P1_UP, None] # Joust specific

        # TODO normalize below to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.observation_space = spaces.Box(low=0, high=255, shape=(JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), dtype=np.uint8) 

        self.render_mode = None 
        self.reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))


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
        
        # actions reset the mame emulation
        self._soft_reset()
        sleep(JoustEnv.BOOT_SECONDS) # wait for game to boot up 
        self._ready_up() # actions to start the game
        self._pause()
        self._throttled_off()
        
        # re-cache last score and lives for both players
        self.last_score = {1: 0, 2: 0}
        self.last_lives = {1: 3, 2: 3}
        
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

    # def _queue_command(self, command):
    #     # Adds semi-colon delimited Lua code to the command queue for execution one frame at a time 
    #     self.mame.execute( dedent(f"""
    #         commands = commands .. '{command}' .. ';'
    #     """))

    def _ready_up(self):
        # Insert a coin
        self._send_input(JoustEnv.COIN1) # Joust specific
        # TODO second input won't work due to "if not running withint _send_input"
        # Press start button
        self._send_input(JoustEnv.P1_START) # Joust specific

    def _wait_frames(n):
        # wait timespan of n frames
        sleep(n/JoustEnv.FPS) 

    def _send_input(self, port_field):
        # takes a (port,field) input tuple e.g. (":IN2","Coin 1") and simulates that user input 
        # these are rom specific and can be found with _get_inputs() 
        port = port_field[0]
        field = port_field[1]
        # TODO : better way?
        self.mame.execute( dedent(f""" 
            sub=emu.add_machine_frame_notifier( function()
                if not running then
                    running=true
                    print( manager.machine.screens[':screen']:frame_number()) 
                    manager.machine.ioport.ports['{port}'].fields['{field}']:set_value(1)
                    emu.wait_next_frame() -- seems to need 2 frames between
                    emu.wait_next_frame() 
                    print( manager.machine.screens[':screen']:frame_number()) 
                    manager.machine.ioport.ports['{port}'].fields['{field}']:set_value(0)
                    sub:unsubscribe()
                    running=false
                end
            end)
        """))

    def _send_action(self, action):
        # takes a Gym action index and simulates that user input
        action_input = self.actions[action]
        if action_input is not None:
            self._send_input(self.actions[action])

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

    def _step(self):
        ret = self.mame.execute("emu.step()")
        if ret != b'OK': raise RuntimeError(ret)

    def _soft_reset(self):
        ret = self.mame.execute("return manager.machine:soft_reset()")
        if ret != b'OK': raise RuntimeError(ret)

    def _get_screen_size(self):
        # TODO can't count ':screen' always being the key for every rom
        result = self.mame.execute("s=manager.machine.screens[':screen']; return s.width .. 'x' .. s.height")
        width, height = map(int, result.decode().split('x'))
        return width, height

    def _get_frame_number(self):
        result = self.mame.execute("s=manager.machine.screens[':screen']; return s:frame_number()")
        return int(result.decode())

    def _get_pixels(self):
        return self.mame.execute("s=manager.machine.screens[':screen']; return s:pixels()")

    def _get_observation(self):
        pixels = self._get_pixels()
        # trim any "footer" data beyond pixel values of JoustEnv.HEIGHT * JoustEnv.WIDTH * 4
        pixels = pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4]
        # unflatten bytes into row,col,channel format; keep all rows and cols, but transform 'ABGR' to RGB  
        observation = np.frombuffer(pixels[:JoustEnv.HEIGHT * JoustEnv.WIDTH * 4], 
                                    dtype=np.uint8).reshape((JoustEnv.HEIGHT, JoustEnv.WIDTH, 4))[:,:,2::-1]
        return observation

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
        reward = score_diff / JoustEnv.MAX_SCORE_DIFF  # Normalize score difference
        
        # Penalty for losing lives
        if lives_diff < 0:
            reward -= 1.0  # Penalty for each life lost
        
        return reward

    def _check_done(self, player=1):
        current_lives = self._get_lives(player)
        return current_lives == 0  # Game is over when player has no lives left

    # def _init_command_queue(self):
    #     # preps the MAME client session to enable issuing a sequence of Lua instructions over successive frames 
    #     self.mame.execute( dedent(r"""
    #         commands = '' -- persistant global string takes semicolon-delimited Lua code for execution in turn each frame 
    #         errors = '' -- persistant global string holds semicolon-delimited error messages from Lua code execution
    #         cmdQ = {} 
    #         process_commands_sub = emu.add_machine_frame_notifier(
    #             function()
    #                 if (string.len(commands) > 0) then -- Check global commands string for new commands
    #                     for cmd in string.gmatch(commands, '[^;]+') do  -- Split command string by ';' and iterate over each part
    #                         if string.len(cmd) > 0 then  -- If the command is not empty
    #                             cmdFn = load(cmd);  -- Create a function for the command 
    #                             table.insert(cmdQ, cmdFn);  -- Add it the quene for execution
    #                         end
    #                     end
    #                     commands = ''  -- Clear the commands string for the next frame
    #                     errors = '' -- Clear errors 
    #                 end
    #                 if #cmdQ > 0 then  -- If there are commands in the queue
    #                     local success, error_msg = pcall(cmdQ[1])  -- Execute the next command function in the queue with error checking
    #                     if not success then
    #                         errors = errors .. error_msg .. ';'  -- Add error message to errors string
    #                         cmdQ={} -- Clear the command queue after error
    #                     end
    #                     table.remove(cmdQ,1);  -- remove the function from the table after executing it
    #                 end
    #             end
    #         )
    #     """))


    # def _get_rom_inputs(self):
    #     # utility fn to get all available "port" and "field" input codes for arbitraty MAME roms 
    #     lua_script = dedent("""
    #     function serialize(obj)
    #         local items = {}
    #         for k, v in pairs(obj) do
    #             if type(v) == "table" then
    #                 items[#items+1] = string.format("%q:{%s}", k, serialize(v))
    #             else
    #                 items[#items+1] = string.format("%q:%q", k, tostring(v))
    #             end
    #         end
    #         return "{" .. table.concat(items, ",") .. "}"
    #     end

    #     function get_inputs()
    #         local inputs = {}
    #         for port_name, port in pairs(manager.machine.ioport.ports) do
    #             inputs[port_name] = {}
    #             for field_name, field in pairs(port.fields) do
    #                 inputs[port_name][field_name] = field.mask
    #             end
    #         end
    #         return serialize(inputs)
    #     end

    #     return get_inputs()
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