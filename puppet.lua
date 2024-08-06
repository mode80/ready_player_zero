emu.pause() -- Pause the game while we wait for a client  
manager.machine.video.throttled = false -- Disable frame limiting

local sock = emu.file("rwc") 
sock:open("socket.127.0.0.1:1942")
emu.print_info("Listening on port 1942")
screen = manager.machine.screens[":screen"]

function handle_command(command)
    if command == "FRAME_NUMBER" then
        sock:write(string.pack(">I4", screen:frame_number())) 

    elseif command == "STEP" then
        emu.step() -- progress the game one frame 
        sock:write(string.pack(">I4", screen:frame_number())) -- Send new frame number as confirmation

    elseif command == "PIXELS_BYTES" then
        local byte_count = #screen:pixels()
        sock:write(string.pack(">I4", byte_count))

    elseif command == "SCREEN_SIZE" then
        local width, height = screen.width, screen.height
        sock:write(string.pack(">I4I4", width, height))
        
    elseif command == "PIXELS" then
        local pixels = screen:pixels()
        sock:write(pixels)

    else 
        emu.print_error("Unexpected command from client: " .. command)
    end
end

function per_frame()
    local command = sock:read(100) -- Check if there's a message from the client 
    if #command > 0 then  
        emu.print_debug("Received command: " .. command)
        handle_command(command)
    end
end

emu.register_frame_done(per_frame)

-- ./mamed -window -autoboot_script puppet.lua -autoboot_delay 1 joust 
-- echo -n "REQUEST" | nc localhost 1942 

