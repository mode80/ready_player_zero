#%%
import socket
import select
import time
import argparse

import socket

def getall(sock, buffer_size=1024):
    """ Get all inbound data from socket without a timeout. """
    data = bytearray()
    sock.setblocking(0)  # Set socket to non-blocking mode

    while True:
        try:
            part = sock.recv(buffer_size)
            if not part:
                break  # No more data
            data.extend(part)
        except BlockingIOError:
            break  # No more data available

    return bytes(data)

def cmd(cmd: str, sock): 
    """ Send command to the server and get response. """
    sock.sendall(cmd.encode('utf-8')) 
    data = getall(sock)
    return data.decode()

def main():
    # take in command line any args 
    parser = argparse.ArgumentParser(description='Socket client.')
    parser.add_argument('--hostname', type=str, default='127.0.0.1', help='Server hostname')
    parser.add_argument('--port', type=int, default=1942, help='Server port')
    args = parser.parse_args()

    # Create a socket object and connect to the server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET: IPv4, SOCK_STREAM: TCP
    sock.connect((args.hostname, args.port))
    
    # Issue commands to the server
    while True:
        response = cmd('FRAME_NUMBER', sock) 
        print(response)
        response = cmd('STEP', sock) 
    
    # Close the socket
    sock.close()


#blahablah
if __name__ == "__main__":
    main()

# %%
