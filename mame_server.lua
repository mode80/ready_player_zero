---------------------------------------------------
-- MAME Lua socket server for remote console access
---------------------------------------------------

-- Configuration
local config = {
    port = 1942,
    host = "127.0.0.1"
}

-- executes Lua code in the MAME object environment and return the result
local function execute_lua(code)
    local func, err = load(code) 
    if func then
        local status, result = pcall(func)
        if status then
            if result == nil then
                return "OK" -- Command executed successfully (no return value)
            else
                return result -- tostring(result)?
            end
        else
            return "Runtime error: " .. tostring(result)
        end
    else
        return "Syntax error: " .. tostring(err)
    end
end

-- checks for inbound Lua commands after each frame
local function runs_per_frame()
    local command = sock:read(1024)
    if #command > 0 then
        if command:sub(1,4) == "quit" then -- Client is intentionally disconnecting
            sock:write("\n") -- Send a newline to acknowledge the command
            emu.print_info("Closing Server connection")
            sock:close() -- Close this connection
            sock:open("socket." .. config.host .. ":" .. config.port) -- Reopen a socket for next
            emu.print_info("MAME listening on port " .. config.port)
        else
            -- Process regular commands
            -- emu.print_debug("Command:")
            emu.print_debug(command)
            local result = execute_lua(command)
            if type(result) == "boolean" then result = tostring(result) end
            sock:write(result .. "\n")
        end
    end
end

-- cue up the socket
sock = emu.file("rwc") 
sock:open("socket." .. config.host .. ":" .. config.port)
emu.print_info("MAME listening on port ".. config.port)

emu.register_frame_done(runs_per_frame)

-- emu.add_machine_stop_notifier(function() sock:close(); print("Socket closed"); end)


