import socket
import logging

class MAMEConsoleInterface:
    def __init__(self, host='127.0.0.1', port=1942):
        self.host = host
        self.port = port
        self.sock = None
        logging.basicConfig(level=logging.INFO)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        logging.info(f"Connected to {self.host}:{self.port}")

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
            logging.info("Socket closed")

# Usage example
if __name__ == "__main__":

    mame = MAMEConsoleInterface()

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
    # print(result)

    # Send an input
    result = mame.execute("i=manager.machine.input; return i:code_pressed(i:code_from_token('P1_BUTTON1'))")
    print(f"Button press result: {result}")

    # Reset the game
    # result = mame.execute("manager.machine:soft_reset()")
    # print(f"Reset game result: {result}")

    mame.close()
