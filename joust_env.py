#%%#####################################################################################################################
from random import randint
from time import sleep
import gymnasium as gym
import numpy as np
import os, subprocess, struct, socket
import matplotlib.pyplot as plt


#%%#####################################################################################################################

class MinJoustEnv(gym.Env): # Minimalist Joust Environment

    SOCKET_PORT = 1942
    FRAMES_PER_STEP = 4 # Joust inputs can take 4 frames to affect the display output, so we don't sample faster than this

    THROTTLED = False 
    WIDTH, HEIGHT = 292, 240 # screen pixel dimensions  
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. 
    READY_UP_FRAMES = 200
    MAME_EXE = '' # full path to mame executable if not './mame'. it must have access to joust rom

    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
 
    init_inputs_lua = ( # wait, waitup and, swait each work better in different contexts when contructing inputs
    """ 
        wait   = function() emu.wait_next_frame() end;
        waitup = function() emu.wait_next_update() end;
        swait  = function() emu.step(); emu.wait_next_frame() end; 
        iop    = manager.machine.ioport.ports; 
        inp1   = iop[':INP1'].fields; 
        in2    = iop[':IN2'].fields; 
        in0    = iop[':IN0'].fields; 
        coin   = function(v)   in2['Coin 1']:set_value(v) end; 
        start  = function(v)   in0['1 Player Start']:set_value(v); end; 
        flap   = function(v)   inp1['P1 Button 1']:set_value(v) end; 
        left   = function()    inp1['P1 Left']:set_value(1); inp1['P1 Right']:set_value(0) end; 
        right  = function()    inp1['P1 Right']:set_value(1); inp1['P1 Left']:set_value(0) end; 
        center = function()    inp1['P1 Left']:set_value(0); inp1['P1 Right']:set_value(0) end; 
    """)

    COIN_TAP    = "coin(1); wait(); wait(); coin(0); " 
    START_TAP   = "start(1); wait(); wait(); start(0); "
    COIN_START  = COIN_TAP + START_TAP

    FLAP        = "flap(1); swait(); flap(0); " 
    # flap animation is 69 steps() after flap(1) when there is no flap(0) release 
    # otherwise 37 frames after flap release. animation starts on 3rd step after flap(1)   

    LEFT        = "left(); "
    RIGHT       = "right(); "
    CENTER      = "center(); "

    LEFT_FLAP   = LEFT + FLAP
    RIGHT_FLAP  = RIGHT + FLAP
    CENTER_FLAP = CENTER + FLAP

    # advanced actions for harder-to-learn fine control over flap press and release
    FLAP_ON         = "flap(1); "
    FLAP_OFF        = "flap(0); "
    LEFT_FLAP_ON    = "left(); flap(1); "
    RIGHT_FLAP_ON   = "right(); flap(1); "

    # Define action space based on available inputs
    actions = [CENTER, FLAP, LEFT, RIGHT] # simplest - no control over flap release timing or simultaneous flap-left,right
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, CENTER_FLAP] # more complex
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, CENTER_FLAP, FLAP_ON, FLAP_OFF, LEFT_FLAP_ON, RIGHT_FLAP_ON] # most
    action_space = gym.spaces.Discrete(len(actions))  

    # pixels should be be normalized to [-1.0, 1.0] in a wrapper or e.g. CnnPolicy of https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    observation_space = gym.spaces.Box(low=0, high=255, shape=(3,HEIGHT//2,WIDTH//2), dtype=np.uint8)

    render_mode = None 
    reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))

    ####################################################################################################################

    def __init__(self):
        super().__init__()
        #prep minimalist Lua server (via transient lua file for Windows compatibility)
        mini_server = os.path.join(os.path.dirname(__file__), 'mini_server.lua')
        script = ("sock=emu.file('rwc'); " # setup server-side TCP socket
                    f"sock:open('socket.127.0.0.1:{self.SOCKET_PORT}'); " # open TCP port 
                    "on_frame=emu.register_frame_done(function() " # this runs once per frame
                        "local cmd, chunk =''; "  # initialize command string vars
                        "while true do "
                            "chunk=sock:read(4096); " # check for inbound socket content 
                            "cmd=cmd..chunk; " # append to command string
                            "if #chunk < 4096 then break; end; " # if less than full read, assume no more data
                        "end; " 
                        "if #cmd>0 then " # if anything was inbound
                            "print(cmd); "
                            "local ok,res=pcall(load(cmd)); " # run it and capture results
                            # "print(ok, res); "
                            "if res==nil then res=''; end; " # if no results, set to empty string
                            "if type(res)~= 'string' then res=tostring(res); end; " # convert to byte string
                            "sock:write(string.pack('<I4',#res)..res); " # write back results with a 4-byte length prefix
                            # "print('_')"
                        "end; "
                    "end); "
                    f"print('Listening on port {self.SOCKET_PORT}...'); "
                )
        open(mini_server, 'w').write(script)
        # launch MAME running Joust and above Lua server script
        exec = self.MAME_EXE or os.path.join(os.path.dirname(__file__), 'mame', 'mame')
        self.mame = subprocess.Popen(
            [ exec, 'joust', '-console', '-window', '-skip_gameinfo', '-sound', 'none', '-pause_brightness', '1.0', '-background_input', '-autoboot_script', mini_server], 
            cwd=os.path.dirname(exec),
        )
        # try connecting python as a client to the MAME lua server
        self._try_connect() 
        # init Lua environment globals etc from here
        self.init_globals_lua = (""" 
            s = manager.machine.screens:at(1); 
            mem = manager.machine.devices[':maincpu'].spaces['program'] ;  
            vid = manager.machine.video;  
            get8 = function(a) return mem:read_u8(a); end; 
            get32 = function(a) return mem:read_u32(a); end; 
            pack = string.pack; 
            printt = function(t) for k,v in pairs(t) do print(k,v) end end; 
        """)
        # init vars for tracking "last_state" values
        self.last_score, self.last_lives = 0,0 
        self.pixel_history = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._run_lua(self.init_globals_lua)
        self._run_lua(self.init_inputs_lua) # init Lua for inputs
        # self._init_frame_debug()

            
    ####################################################################################################################

    def reset(self, seed=None, options=None):
        # override the mandatory gym.Env reset() method for Joust
        super().reset(seed=seed)
        self._run_lua("vid.throttled = false; ") # speed up coming boot sequence
        self._run_lua("manager.machine:soft_reset(); ")# soft reset mame. (lua event loop survives but connection doesn't)
        self._try_connect()
        self._run_lua( self.init_inputs_lua ) 
        self._run_lua( self.init_globals_lua ) 
        sleep(2) # let boot sequence playout while still unthrolled
        self._run_lua( f"vid.throttled = {str(self.THROTTLED).lower()}; " )# Set throttle back to default
        self._run_lua(self.COIN_START) # Insert coin and start game
        self._run_lua( f"for i=1,{self.READY_UP_FRAMES} do wait() end; emu.step(); ")# Wait for play to start
        pixel_history, _, _, _, info = self.step() # step with no action to get initial state and set initial last_values
        self.pixel_history[1:,:,:] = pixel_history[0,:,:] # copy this 1st frame to 'historical' frames to indicate no motion)
        return self.pixel_history, info

    ####################################################################################################################

    def step(self, action_idx=None):
        # overrides the mandatory gym.Env step() method for Joust
        input_lua = self.actions[action_idx]    if action_idx is not None else ''   
        lua = ( input_lua + 
            " swait(); " * self.FRAMES_PER_STEP +
            f"local lives, score = get8({self.P1_LIVES_ADDR}), get32({self.P1_SCORE_ADDR}); " # extract lives, score from memory
            "return pack('>B I4', lives, score)..s:pixels(); ") # bitpak and return lives, score and screen pixels
        response = self._run_lua(lua, expected_len=5+240*292*4) # 5 bytes for lives, score; 240*292*4 for pixels
        lives, score = struct.unpack('>B I', response[:5]) # unpack lives, score from first 1, then next 4 bytes respectively
        # unflatten bytes into row,col,channel format; # keep all rows and cols, but transform 'ABGR' to RGB, 
        pixels = np.frombuffer(response[5:], dtype=np.uint8).reshape((240, 292, 4))[:,:,2::-1] 
        # downsample to half the resolution
        pixels = pixels[::2, ::2, :]
        # Move chanels to first dimension, per convention
        pixels = np.moveaxis(pixels, -1, 0) #shape is now (1, 120, 146)
        # Convert to grayscale 
        pixels = np.mean(pixels, axis=0).astype(np.uint8) 
        # Calculate reward, done status, info then return with observation 
        score_diff = score - self.last_score
        lives_diff = lives - self.last_lives
        reward = score_diff / self.MAX_SCORE_DIFF  # Normalize score difference
        if lives_diff < 0: reward = -1.0  # Penalty for losing a life
        done = (lives == 0)
        # track last values 
        self.pixel_history = np.roll(self.pixel_history, 1, axis=0)  # cycle 'channel' from [old, older, oldest] to [oldest, old, older]
        self.pixel_history[0] = pixels # replace oldest with current observation
        truncated = False
        info = {'lives': lives, 'score': score}
        return self.pixel_history, reward, done, truncated, info

    ####################################################################################################################

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return np.transpose(self.pixel_history, (1, 2, 0))  # return last 3 frames as a single 'color-coded' image of motion
        elif mode == 'human':
            if not hasattr(self, 'fig'):
                plt.ion()  # Turn on interactive mode
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
                self.fig.suptitle('MinJoustEnv Visualization')
                
            # Display the current frame
            self.ax1.clear()
            self.ax1.imshow(self.pixel_history[0], cmap='gray')
            self.ax1.set_title('Current Frame')
            self.ax1.axis('off')
            
            # Display the motion history
            motion_history = np.transpose(self.pixel_history, (1, 2, 0))
            self.ax2.clear()
            self.ax2.imshow(motion_history)
            self.ax2.set_title('Motion History (Last 3 Frames)')
            self.ax2.axis('off')
            
            # Add text with current game info
            info_text = f"Lives: {self.last_lives}\nScore: {self.last_score}"
            self.fig.text(0.02, 0.02, info_text, verticalalignment='bottom')
            
            plt.draw()
            plt.pause(0.001)  # Small pause to update the plot


    def _init_frame_debug(self):
        # optional code for printing frame number and pause count as they happen
        frame_debug_lua = (  
            "last_frame = 0; ponce = 0; "
            "on_frame_debug=emu.add_machine_frame_notifier( function() " 
                "this_frame = s:frame_number() ; "
                "is_paused = (this_frame == last_frame) ; "
                "if is_paused then ponce = ponce + 1 else ponce = 0 end ; "
                "last_frame = this_frame ; "
                "print(this_frame..':'..ponce) ; "
                "end)"
        )
        self._run_lua(frame_debug_lua)


    def _try_connect(self, tries=200, per_try_delay=0.1):
        # connect python gym env as client to MAME Lua server
        for _ in range(tries): # try to connect to MAME for 20 secs or so while it starts up
            try: 
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # setup client side TCP socket 
                self.sock.connect(('127.0.0.1', self.SOCKET_PORT)); break  # establish connection 
            except: sleep(per_try_delay); continue
    

    def _run_lua(self, lua, expected_len=None):
        # sends Lua code for execution and waits for its (length prefixed) response once it completes
        self.sock.sendall(lua.encode())
        len = expected_len # init 
        if len is None: # if expected response length was unsuplied we need to spend a rountrip to check it
            len_encoded = self.sock.recv(4) # this blocks until server code runs fully and 4-byte encoded len value is received
            len = struct.unpack('<I', len_encoded)[0] # this first recv will be how many bytes the coming main response will be
        else:
            len = expected_len + 4 # extra 4 bytes accomodates the length prefix
        ret = self.recv_all(len) # this blocks execution until all the bytes of 'len' length are received or errors trying
        return ret if expected_len is None else ret[4:] # strip off the 4-byte length prefix before returning as needed
    

    def recv_all(self, expected_len):
        # receive all expected bytes of expected_len from the socket
        data = b''
        remaining = expected_len
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk: raise ConnectionError("Could not receive expected data length")
            data += chunk
            remaining -= len(chunk)
        return data

        
#%%#######################################################################################################################

# Example usage
if __name__ == "__main__":
    env = MinJoustEnv()
    env.reset()

    for _ in range(10000): 
        action = env.action_space.sample()  # Random action
        # action = [0,0,2,2,3,3,1,2,3,4,5,6][randint(0,11)]
        # 10-steps to reverse direction in place. 5 steps after the 1st animates
        # action = [2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2][_ % 64] 
        # action =   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][_ % 50] 
        # action =   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][_ % 24] 
        observation, reward, done, truncated, info = env.step(action)
        env.render(mode:='human')
        
        if done or truncated:
            observation, info = env.reset()
        
# %%
