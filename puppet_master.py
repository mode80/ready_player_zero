import socket
import struct

class MAMEInterface:
    def __init__(self, host='127.0.0.1', port=1942):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def send_command(self, command):
        self.sock.sendall(command.encode())

    def receive_exact(self, size):
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data += chunk
        return data

    def get_frame_number(self):
        self.send_command("FRAME_NUMBER")
        return struct.unpack('>I', self.receive_exact(4))[0]

    def step_frame(self):
        self.send_command("STEP")
        return struct.unpack('>I', self.receive_exact(4))[0]  # Returns new frame number

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

    def close(self):
        if self.sock:
            self.sock.close()

# Usage example
if __name__ == "__main__":
    mame = MAMEInterface()
    try:

        mame.connect()

        print(f"Screen size: {mame.get_screen_size()}")

        bytes_len = mame.get_pixels_bytes()
        print(f"Pixels bytes len: {bytes_len}")
        
        print(f"Current frame: {mame.get_frame_number()}")

        new_frame = mame.step_frame()
        print(f"Stepped to frame: {new_frame}")
        
        # Example of repeated pixel fetching
        for _ in range(5):  # Simulate fetching 5 frames
            pixels = mame.get_pixels(bytes_len)
            print(f"Received pixel data of size: {len(pixels)} bytes")
            new_frame = mame.step_frame()
            print(f"Stepped to frame: {new_frame}")
    finally:
        mame.close()