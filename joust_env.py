from time import sleep, time
import gymnasium as gym
import numpy as np
import os, subprocess, struct, socket, tempfile


class MinJoustEnv(gym.Env): # Minimalist Joust Environment

    THROTTLED = False
    SOCKET_PORT = 1942

    WIDTH, HEIGHT = 292, 240 # screen pixel dimensions  
    MAX_SCORE_DIFF = 3000.0 # Maximum score difference for a single step. 
    READY_UP_FRAMES = 200
    MAME_EXE = '' # full path to mame executable if not './mame'. it must have access to joust rom

    P1_LIVES_ADDR = 0xA052
    P1_SCORE_ADDR = 0xA04C
    # P2_LIVES_ADDR = 0xA05C
    # P2_SCORE_ADDR = 0xA058
 
    init_inputs_lua = (
        "wait   = emu.wait_next_frame; "
        "iop    = manager.machine.ioport.ports; "
        "inp1   = iop[':INP1'].fields; "
        "in2    = iop[':IN2'].fields; "
        "in0    = iop[':IN0'].fields; "
        "coin   = function(v)   in2['Coin 1']:set_value(v) end; "
        "start  = function(v)   in0['1 Player Start']:set_value(v); end; "
        "flap   = function(v)   inp1['P1 Button 1']:set_value(v) end; "
        "left   = function()    inp1['P1 Left']:set_value(1); inp1['P1 Right']:set_value(0) end; "
        "right  = function()    inp1['P1 Right']:set_value(1); inp1['P1 Left']:set_value(0) end; "
        "center = function()    inp1['P1 Left']:set_value(0); inp1['P1 Right']:set_value(0) end; "
    )

    COIN_TAP    = "coin(1); wait(); wait(); coin(0); "
    START_TAP   = "start(1); wait(); wait(); start(0); "

    FLAP        = "flap(1); wait(); flap(0); "

    LEFT        = "left(); "
    RIGHT       = "right(); "
    CENTER      = "center(); "

    LEFT_FLAP   = LEFT + FLAP
    RIGHT_FLAP  = RIGHT + FLAP
    CENTER_FLAP = CENTER + FLAP

    # advanced actions for harder-to-learn fine control over flap press and release
    FLAP_PRESS  = "flap(1); "
    FLAP_RELEASE= "flap(0); "
    LEFT_FLAP_PRESS = "left(); flap(1);"
    RIGHT_FLAP_PRESS = "right(); flap(1);"

    # Define action space based on available inputs
    actions = [CENTER, FLAP, LEFT, RIGHT] # simplest - no control over flap release timing or simultaneous flap-left,right
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, FLAP_CENTER] # more complex
    # actions = [CENTER, FLAP, LEFT, RIGHT, LEFT_FLAP, RIGHT_FLAP, FLAP_CENTER, FLAP_PRESS, FLAP_RELEASE, LEFT_FLAP_PRESS, RIGHT_FLAP_PRESS] # most
    action_space = gym.spaces.Discrete(4)  

    # pixels normalized to [0, 1] per https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    observation_space = gym.spaces.Box(low=0, high=1.0, shape=(HEIGHT,WIDTH), dtype=np.float32)

    render_mode = None 
    reward_range = (-1.0, 1.0) # (-float("inf"), float("inf"))

    def __init__(self):
        super().__init__()

        #prep minimalist Lua server (via transient lua file for Windows compatibility)
        # temp = tempfile.NamedTemporaryFile(mode='w', suffix='.lua')  
        mini_server = os.path.join(os.path.dirname(__file__), 'mini_server.lua')
        script = ("sock=emu.file('rwc'); " # setup server-side TCP socket
                    "sock:open('socket.127.0.0.1:1942'); " # same hard-baked port used below
                    "on_frame=emu.add_machine_frame_notifier(function() " # this runs once per frame
                    "cmd=sock:read(4096);"
                    "ok,res=pcall(load(cmd)); res=res or ''; " # read socket content and execute it. 4096 bytes should be enough? 
                    "sock:write(string.pack('<I4',#res)..res)" # write back results with 4-byte length prefix
                    "end)"
                    "print('listening...')"
                )
        open(mini_server, 'w').write(script)

        # launch MAME running Joust and Lua server script
        exec = self.MAME_EXE or os.path.join(os.path.dirname(__file__), 'mame', 'mame')
        self.mame = subprocess.Popen(
            [ exec, 'joust', '-console', '-window', '-skip_gameinfo', '-pause_brightness', '1.0', '-autoboot_script', mini_server], 
            cwd=os.path.dirname(exec),
            # shell=True,
        )
        # os.chdir(os.path.dirname(exec)) 
        # subprocess.run( exec + ' joust -console -window -skip_gameinfo -pause_brightness 1.0 -autoboot_script ' + mini_server , start_new_session=True) 

        # connect python gym env as client
        for _ in range(200): # try to connect to MAME for 20 secs or so while it starts up
            try: 
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # setup client side TCP socket 
                self.sock.connect(('127.0.0.1', self.SOCKET_PORT)); break  # establish connection 
            except: sleep(0.1)

        # init Lua environment
        init_lua_globals = ( # some handy global shortcuts
            "s = manager.machine.screens:at(1); "
            "mem = manager.machine.devices[':maincpu'].spaces['program'] ; " 
            "printt = function(t) for k,v in pairs(t) do print(k,v) end end; "
        )
        self._send_lua(init_lua_globals)
        self._send_lua(self.init_inputs_lua) # init Lua for inputs
        self._init_frame_debug()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        lua_code = (
            "manager.machine:soft_reset(); "# soft reset mame. (lua env survives)
            "manager.machine.video.throttled = false; " # speed up boot sequence
            f"while mem:read_u8({self.P1_LIVES_ADDR}) ~= 3 do emu.wait_next_frame() end; "# Wait for reboot done (checking P1 lives) TODO
            f"{self.COIN_TAP} {self.START_TAP} "# Insert coin and start game
            f"for i=1,{self.READY_UP_FRAMES} do emu.wait_next_frame() end; "# Wait for play to start
            f"manager.machine.video.throttled = {str(self.THROTTLED).lower()}; "# Set throttle back to default
        )
        self._send_lua(lua_code)
        observation, _, _, _, info = self.step() # step with no action to get initial state
        return observation, info


    def step(self, action_idx):

        input_lua=''
        if action_idx is not None: 
            input_lua = self.actions[action_idx]
            print(input_lua) # debug

        lua_code = (
            input_lua + "; emu.wait_next_frame(); "
            f"local lives = mem:read_u8({self.P1_LIVES_ADDR}); " # read player lives from memory
            f"local score = mem:read_u32({self.P1_SCORE_ADDR}); " # read player score from memory
            "return string.pack('>B I4', lives, score) .. s:pixels()" # bitpak and return lives, score and screen pixels
        )

        response = self._send_lua(lua_code)
        lives, score = struct.unpack(">BI", response[:5])

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

    def close(self):
        if self.sock: self.sock.close()
        if self.mame: self.mame.kill()  # try terminate()?

    def _send_lua(self, lua_code):
        self.sock.sendall(lua_code.encode())
        length = struct.unpack('<I', self.sock.recv(4))[0]
        return self.sock.recv(length)
    
    def _init_frame_debug(self):
        # optional code for printing frame number and pause count as they happen
        frame_debug_lua = (  
            "last_frame = 0; ponce = 0;"
            "on_frame_debug=emu.add_machine_frame_notifier(function() " 
                "this_frame = s:frame_number() ; "
                "is_paused = (this_frame == last_frame) ; "
            "if is_paused then ponce = ponce + 1 else ponce = 0 end ; "
            "print(this_frame..':'..ponce) ; "
        )
        self._send_lua(frame_debug_lua)
        

# Example usage
if __name__ == "__main__":
    env = MinJoustEnv()
    env.reset()

    for _ in range(10000): 
        # action = env.action_space.sample()  # Random action
        # action = [0,0,2,2,3,3,1,2,3,4,5,6][randint(0,11)]
        # 10-steps to reverse direction in place. 5 steps after the 1st animates
        # action = [2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,2,2,2,2,2,2][_ % 64] 
        action =   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][_ % 50] 
        observation, reward, done, truncated, info = env.step(action)
        # sleep(.5)
        
        if done or truncated:
            observation, info = env.reset()
        
    env.close()
