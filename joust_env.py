from time import sleep, time
import gymnasium as gym
import numpy as np
import os, subprocess, struct, socket, tempfile


###########################################################################################################################

class MinJoustEnv(gym.Env): # Minimalist Joust Environment

    SOCKET_PORT = 1942

    THROTTLED = False
    WIDTH, HEIGHT = 292, 240 # screen pixel dimensions  
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. 
    READY_UP_FRAMES = 200
    MAME_EXE = '' # full path to mame executable if not './mame'. it must have access to joust rom

    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
    # P2_LIVES_ADDR = 0xA05C
    # P2_SCORE_ADDR = 0xA058
 
    init_inputs_lua = ("""
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
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, FLAP_CENTER] # more complex
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, FLAP_CENTER, FLAP_PRESS, FLAP_RELEASE, LEFT_FLAP_PRESS, RIGHT_FLAP_PRESS] # most
    action_space = gym.spaces.Discrete(4)  

    # pixels normalized to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    observation_space = gym.spaces.Box(low=0, high=1.0, shape=(HEIGHT,WIDTH), dtype=np.float32)

    render_mode = None 
    reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))

    ###########################################################################################################################

    def __init__(self):
        super().__init__()

        #prep minimalist Lua server (via transient lua file for Windows compatibility)
        # temp = tempfile.NamedTemporaryFile(mode='w', suffix='.lua')  
        mini_server = os.path.join(os.path.dirname(__file__), 'mini_server.lua')
        script = ("sock=emu.file('rwc'); " # setup server-side TCP socket
                    "sock:open('socket.127.0.0.1:1942'); " # same hard-baked port used below
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
                            "sock:write(string.pack('<I4',#res)..res); " # write back results with 4-byte length prefix
                            # "print('_')"
                        "end; "
                    "end); "
                    "print('listening...'); "
                )
        open(mini_server, 'w').write(script)

        # launch MAME running Joust and Lua server script
        exec = self.MAME_EXE or os.path.join(os.path.dirname(__file__), 'mame', 'mame')
        self.mame = subprocess.Popen(
            [ exec, 'joust', '-console', '-window', '-skip_gameinfo', '-pause_brightness', '1.0', '-background_input', '-autoboot_script', mini_server], 
            cwd=os.path.dirname(exec),
        )

        self._try_connect()

        # init Lua environment
        self.init_globals_lua = (""" 
            s = manager.machine.screens:at(1); 
            mem = manager.machine.devices[':maincpu'].spaces['program'] ;  
            vid = manager.machine.video;  
            get8 = function(a) return mem:read_u8(a); end; 
            get32 = function(a) return mem:read_u32(a); end; 
            pack = string.pack; 
            printt = function(t) for k,v in pairs(t) do print(k,v) end end; 
        """)
        self._run_lua(self.init_globals_lua)
        self._run_lua(self.init_inputs_lua) # init Lua for inputs
        self._init_frame_debug()
            
    ###########################################################################################################################

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._run_lua("vid.throttled = false; ") # speed up coming boot sequence
        self._run_lua("manager.machine:soft_reset(); ")# soft reset mame. (lua event loop survives but connection doesn't)

        self._try_connect()

        self._run_lua( self.init_inputs_lua )  # TODO needed?
        self._run_lua( self.init_globals_lua ) 
        sleep(2) # TODO needed?
        self._run_lua( f"vid.throttled = true ") #{str(self.THROTTLED).lower()}; " )# Set throttle back to default
        self._run_lua(self.COIN_START) # Insert coin and start game
        self._run_lua( f"for i=1,{self.READY_UP_FRAMES} do wait() end; emu.step(); ")# Wait for play to start

        self.last_score, self.last_lives = 0,0 # re-init 

        observation, _, _, _, info = self.step() # step with no action to get initial state
        return observation, info

    ###########################################################################################################################

    def step(self, action_idx=None):

        input_lua = self.actions[action_idx]    if action_idx is not None else ''   
        lua = ( 
            input_lua + "; swait(); "
            f"local lives, score = get8({self.P1_LIVES_ADDR}), get32({self.P1_SCORE_ADDR}); " # extract lives, score from memory
            "return pack('>B I4', lives, score)..s:pixels(); ") # bitpak and return lives, score and screen pixels
        response = self._run_lua(lua, expected_len=5+240*292*4) # 5 bytes for lives, score; 240*292*4 for pixels
        lives, score = struct.unpack('>B I', response[:5]) # unpack lives, score from first 1, then next 4 bytes respectively

        # unflatten bytes into row,col,channel format; # keep all rows and cols, but transform 'ABGR' to RGB, 
        observation = np.frombuffer(response[5:], dtype=np.uint8).reshape((240, 292, 4))[:,:,2::-1] 
        # then convert to grayscale and normalize to range of [0, 1]
        observation = np.mean(observation, axis=-1) / 255.0

        # Calculate reward
        score_diff = score - self.last_score
        lives_diff = lives - self.last_lives
        reward = score_diff / self.MAX_SCORE_DIFF  # Normalize score difference
        if lives_diff < 0: reward = -1.0  # Penalty for losing a life
        
        # Check if done
        done = (lives == 0)

        # Update last score and lives
        self.last_score, self.last_lives = score, lives

        info = {'lives': lives, 'score': score}
        return observation, reward, done, False, info

    ###########################################################################################################################

    def close(self):
        if self.sock: self.sock.close()
        if self.mame: self.mame.kill()  # try terminate()?

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


    def _try_connect(self):
        # connect python gym env as client
        for _ in range(200): # try to connect to MAME for 20 secs or so while it starts up
            try: 
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # setup client side TCP socket 
                self.sock.connect(('127.0.0.1', self.SOCKET_PORT)); break  # establish connection 
            except: sleep(0.1)
    
    def _run_lua(self, lua, expected_len=None):
        # sends lua for execution and waits for the (length prefixed) response once it completes
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

        
###########################################################################################################################

# Example usage
if __name__ == "__main__":
    env = MinJoustEnv()
    env.reset()

    for _ in range(10000): 
        # action = env.action_space.sample()  # Random action
        # action = [0,0,2,2,3,3,1,2,3,4,5,6][randint(0,11)]
        # 10-steps to reverse direction in place. 5 steps after the 1st animates
        action = [2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2][_ % 64] 
        # action =   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][_ % 50] 
        observation, reward, done, truncated, info = env.step(action)
        # sleep(.5)
        
        if done or truncated:
            observation, info = env.reset()
        
    env.close()

# TODO commit 58cbda8481a3cd6b7cd57e12a6efe7f6623e8031 might be better since every lua command was synchronous (more consistency)
# so far neither approach makes jump work :/