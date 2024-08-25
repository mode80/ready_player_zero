import socket
import argparse

class MAMEClient:
    def __init__(self, host='127.0.0.1', port=1942):
        self.host = host
        self.port = port
        self.sock = None  

    def connect(self):
        # if self.sock: self.close() # Close existing connection if any
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create TCP socket
        self.sock.connect((self.host, self.port))  # Establish connection to MAME
        print(f"Connected to {self.host}:{self.port}")

    def close(self):
        if not self.sock: return  # Do nothing if not connected
        self.sock.sendall(b'quit\n')  # Special quit command closes the server-side connection 
        self.sock.close()  # Close the socket if it exists
        self.sock = None
        print("Closing Client connection")

    def execute(self, lua_code):
        if not self.sock: raise ConnectionError("Not connected to MAME")
        self.sock.sendall(lua_code.encode() + b'\n')  # Send Lua code with newline delimiter
        response = b''
        while b'\n' not in response:  # Loop until newline is received
            chunk = self.sock.recv(32767)# (65536) # (32767)# (4096)  # Receive data in chunks
            if not chunk: raise ConnectionError("Connection closed while receiving data")
            response += chunk
        return response[:-1]  # Return response without the trailing newline



def main():

    blurb = """
    Connects to a MAME instance that's running the 'mame_server.lua' script, allowing you to send Lua commands remotely.
    For available MAME Lua commands, visit: https://docs.mamedev.org/luascript/index.html 

    Example:

        % ./mame joust -window -autoboot_script ~/mamegym/mame_server.lua 
        MAME listening on 127.0.0.1:1942

        % python ~/mamegym/mame_client.py 
        Connected to 127.0.0.1:1942

        > emu.pause()
        b'OK'

        > a=1; b=2
        b'OK'

        > return a+b
        b'3'

        > quit
    """

    # Set up command line argument parsing
    parser = argparse.ArgumentParser( 
        description=blurb,
        formatter_class=argparse.RawDescriptionHelpFormatter) 
    parser.add_argument("--host", default="127.0.0.1", help="MAME console host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=1942, help="MAME console port (default: 1942)")
    args = parser.parse_args()

    # Control MAME from the command line
    mame = MAMEClient(host=args.host, port=args.port)
    mame.connect()
    while True: # user input loop sends each command to the console and returns output
        lua_code = input("\n> ")
        try:
            result = mame.execute(lua_code)
            print(f"{result}")
        except Exception as e:
            print(f" ERROR: {e}")
        if lua_code.lower().strip() == "quit": break 
    
    mame.close()


def sample_usage():

    mame = MAMEClient()

    mame.connect()

    #Pause the game 
    result = mame.execute("emu.pause()")
    print(f"Paused : {result}")

    # Turn off throttling 
    result = mame.execute("manager.machine.video.throttled = false")
    print(f"Throttle off : {result}")

    # Get screen size
    result = mame.execute("s=manager.machine.screens[':screen']; return s.width .. 'x' .. s.height") # TODO should iterate over all screens
    print(f"Screen size: {result}")

    # Get current frame number
    result = mame.execute("s=manager.machine.screens[':screen']; return s:frame_number()")
    print(f"Current frame: {result}")
    # Step frame
    result = mame.execute("emu.step()")
    print(f"Stepped: {result}")

    # Get pixel data
    result = mame.execute("s=manager.machine.screens[':screen']; return s:pixels()")
    print(result)

    # Detect an input
    result = mame.execute("i=manager.machine.input; return i:code_pressed(i:code_from_token('P1_BUTTON1'))")
    print(f"Button press result: {result}")

    # Reset the game
    result = mame.execute("manager.machine:soft_reset()")
    print(f"Reset game result: {result}")

    mame.close()


if __name__ == "__main__": main()

