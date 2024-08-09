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
                return result 
            end
        else return "Runtime error: " .. tostring(result) end
    else return "Syntax error: " .. tostring(err) end
end

local function listen_for_client()
    if not sock then sock = emu.file("rwc") end
    sock:open("socket." .. config.host .. ":" .. config.port) -- Reopen a socket for next
    emu.print_info("MAME listening on " .. config.host .. ":" .. config.port .. "\n")
end

-- checks for inbound Lua commands after each frame
local function runs_per_frame()
    local command = sock:read(1024)
    if #command > 0 then
        if command:sub(1,4) == "quit" then -- Client is intentionally disconnecting
            sock:write("\n") -- Send a newline to acknowledge the command
            emu.print_info("Closing Server connection")
            sock:close() -- Close this connection
            listen_for_client() -- Start listening for a new client
        else
            -- Process regular commands
            emu.print_debug(command:gsub("\n$", "")) -- debugging output  
            local result = execute_lua(command) -- ! 
            -- result = tostring(result) 
            sock:write(result .. "\n") -- Send the result back to the client
        end
    end
end


listen_for_client()
emu.register_frame_done(runs_per_frame)
