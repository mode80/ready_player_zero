import socket
import argparse

class MAMEConsole:
    def __init__(self, host='127.0.0.1', port=1942):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def execute(self, lua_code):
        self.sock.sendall(lua_code.encode() + b'\n')  # Add newline as a delimiter
        response = b''
        while b'\n' not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            response += chunk
        #  trim the newline "end of transimission" character
        return response[:-1]

    def close(self):
        if self.sock:
            self.sock.close()
            print("Socket closed")

if __name__ == "__main__":

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Connect to MAME remote console")
    parser.add_argument("--host", default="127.0.0.1", help="MAME console host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=1942, help="MAME console port (default: 1942)")
    args = parser.parse_args()

    # Control MAME from the command line
    mame = MAMEConsole(host=args.host, port=args.port)
    mame.connect()

    # run a user input look sending each command to the console and returning the output
    while True:
        lua_code = input("[MAME] ")
        if lua_code.lower() == 'quit':
            break
        try:
            result = mame.execute(lua_code)
            print(f"{result}")
        except Exception as e:
            print(f"ERROR: {e}")

    mame.close()


def sample_usage():

    mame = MAMEConsole()

    mame.connect()

    #Pause the game while we wait for a client  
    result = mame.execute("emu.pause()")
    print(f"Paused : {result}")

    # Turn off throttling 
    result = mame.execute("manager.machine.video.throttled = false")
    print(f"Throttle off : {result}")

    # Get screen size
    result = mame.execute("s=manager.machine.screens[':screen']; return s.width .. 'x' .. s.height") # TODO need to iterate over all screens
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
