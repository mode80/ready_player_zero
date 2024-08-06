
emu.pause() -- Pause the game while we wait for a client  

local sock = emu.file("rwc") 
sock:open("socket.127.0.0.1:1942")
emu.print_info("Listening on port 1942")
screen = manager.machine.screens[":screen"]

function fun()
    local message = sock:read(100) -- Check if there's a message from the client 
    if #message > 0 then  
        -- emu.print_info("Received from client: " .. message)
        if message == "REQUEST" then
            sock:write("Frame number: " .. screen:frame_number() .. "\n") -- Send a message to the Python server
            emu.step() -- progress the game one frame 
        else
            emu.print_debug("Unexpected message from client: " .. message)
        end
    end
end

emu.register_frame_done(fun)

-- Close the socket
-- sock:close()
-- print("Socket closed. Script terminated.")


-- local function wait_to_read(socket, timeout, bytes_at_once)
--     timeout=timeout or 1 -- default timeout 1 second
--     bytes_at_once=bytes_at_once or 100 -- default read 100 bytes at once
--     local data = ""
--     local time = os.time()
--     repeat
--         local res = socket:read(bytes_at_once)
--         data = data .. res
--     until #res == 0 and #data > 0 or time + 1 < os.time()
--     if data:find("ERR", 1, true) then
--         print("Bad RPC reply, " .. data:sub(8) .. "\n")
--     end
--     if #data == 0 then print("timed out waiting for response\n") end
--     return data
-- end

-- ./mamed -window -autoboot_script puppet.lua -autoboot_delay 1 joust 
-- echo -n "REQUEST" | nc localhost 1942 

