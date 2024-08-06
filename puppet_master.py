#%%
import socket
import argparse
import time
import select

def getall(sock, buffer_size=100, timeout=3):
    """ Get all inbound data from socket with a timeout, handling any data size correctly. """
    data = bytearray()
    start_time = time.time()
    while True:
        try:
            ready = select.select([sock], [], [], max(0, timeout - (time.time() - start_time)))
            if ready[0]:
                part = sock.recv(buffer_size)
                if not part:  # Connection closed by remote end
                    break
                data.extend(part)
                if len(part) < buffer_size:  # Less data than buffer_size, likely all received
                    break
            elif time.time() - start_time >= timeout:
                break  # Timeout reached
        except socket.error as e:
            print(f"Socket error: {e}")
            break
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
