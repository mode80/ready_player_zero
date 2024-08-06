import socket
import struct
import logging
import time

class MAMEInterface:
    def __init__(self, host='127.0.0.1', port=1942):
        self.host = host
        self.port = port
        self.sock = None
        logging.basicConfig(level=logging.INFO)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        logging.info(f"Connected to {self.host}:{self.port}")

    def send_command(self, command):
        self.sock.sendall(command.encode())
        logging.debug(f"Sent command: {command}")

    def receive_exact(self, size):
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data += chunk
        logging.debug(f"Received data: {data}")
        return data

    def get_frame_number(self):
        self.send_command("FRAME_NUMBER")
        return struct.unpack('>I', self.receive_exact(4))[0]

    def step_frame(self):
        self.send_command("STEP")
        return struct.unpack('>I', self.receive_exact(4))[0]

    def get_screen_size(self):
        self.send_command("SCREEN_SIZE")
        width, height = struct.unpack('>II', self.receive_exact(8))
        return (width, height)

    def get_pixels_bytes(self):
        self.send_command("PIXELS_BYTES")
        bytes_len = struct.unpack('>I', self.receive_exact(4))[0]
        return bytes_len 

    def get_pixels(self, bytes_len):
        self.send_command("PIXELS")
        pixel_data = self.receive_exact(bytes_len) 
        return pixel_data

    def send_action(self, action):
        # Map action to MAME input commands
        # This is an example mapping for Joust
        actions = [
            "P1_LEFT",
            "P1_RIGHT",
            "P1_BUTTON1",
            "P1_LEFT P1_BUTTON1",
            "P1_RIGHT P1_BUTTON1",
            ""  # No-op
        ]
        command = f"INPUT {actions[action]}"
        self.send_command(command)

    def reset_game(self):
        self.send_command("SOFT_RESET")
        # Wait for the game to fully reset
        time.sleep(2)

    def close(self):
        if self.sock:
            self.sock.close()
            logging.info("Socket closed")

# Usage example
if __name__ == "__main__":
    mame = MAMEInterface()
    try:
        mame.connect()
        print(f"Screen size: {mame.get_screen_size()}")
        bytes_len = mame.get_pixels_bytes()
        print(f"Pixels bytes len: {bytes_len}")
        print(f"Current frame: {mame.get_frame_number()}")
        frame_num = mame.step_frame()
        print(f"Stepped to frame: {frame_num}")

        # Example of repeated pixel fetching
        for _ in range(5):  # Reduced to 5 iterations for brevity
            pixels = mame.get_pixels(bytes_len)
            print(f"Received pixel data of size: {len(pixels)} bytes")
            frame_num = mame.step_frame()
            print(f"Stepped to frame: {frame_num}")

        # Example of sending an action
        mame.send_action(0)  # Send P1_LEFT action
        print("Sent P1_LEFT action")

        # Example of resetting the game
        mame.reset_game()
        print("Reset the game")

    finally:
        mame.close()