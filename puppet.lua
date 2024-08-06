-- Configuration
local config = {
    port = 1942,
    host = "127.0.0.1"
}

emu.pause() -- Pause the game while we wait for a client  
manager.machine.video.throttled = false -- Disable frame limiting

local sock = emu.file("rwc") 
sock:open("socket." .. config.host .. ":" .. config.port)
emu.print_info("Listening on port " .. config.port)

local screen = manager.machine.screens[":screen"]

-- Command handlers
local commands = {
    FRAME_NUMBER = function()
        return string.pack(">I4", screen:frame_number())
    end,
    STEP = function()
        emu.step()
        return string.pack(">I4", screen:frame_number())
    end,
    PIXELS_BYTES = function()
        return string.pack(">I4", #screen:pixels())
    end,
    SCREEN_SIZE = function()
        return string.pack(">I4I4", screen.width, screen.height)
    end,
    PIXELS = function()
        return screen:pixels()
    end,
    MARCO = function()
        return "POLO"
    end
}

local function handle_command(command)
    local handler = commands[command]
    if handler then
        local success, result = pcall(handler)
        if success then
            sock:write(result)
        else
            emu.print_error("Error executing command: " .. result)
            sock:write("ERROR")
        end
    else
        emu.print_error("Unknown command: " .. command)
        sock:write("BAD COMMAND")
    end
end

local function per_frame()
    local command = sock:read(100) -- Check if there's a message from the client 
    if #command > 0 then  
        emu.print_debug("Received command: " .. command)
        handle_command(command)
    end
end

emu.register_frame_done(per_frame)

emu.add_machine_stop_notifier(function()
    sock:close()
end)

-- Usage instructions
--  To use this script, start MAME with the following command line:
--      ./mamed -window -autoboot_script puppet.lua -autoboot_delay 1 <game_rom>"
--  As a basic test, this should return "POLO": 
--      printf 'MARCO' | nc localhost 1942"