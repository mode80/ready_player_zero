#%%
import socket

# misc vars
timeout_secs = 5

# Create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET: IPv4, SOCK_STREAM: TCP

# Connect to the server on local computer
sock.connect(('127.0.0.1', 1942))

while True:

    # Send data to the Lua server
    sock.sendall(b'REQUEST') # b: bytes literal 

    # Receive data from the Lua server
    data = sock.recv(1024) 
    print('Received from Lua:', data.decode())


# Close the connection
sock.close()


# %%
