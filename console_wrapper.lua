-- Configuration
local config = {
    port = 1942,
    host = "127.0.0.1"
}

-- emu.pause() -- Pause the game while we wait for a client  
-- manager.machine.video.throttled = false -- Disable frame limiting

local sock = emu.file("rwc") 
sock:open("socket." .. config.host .. ":" .. config.port)
emu.print_info("Listening on port ".. config.port)

-- executes Lua code in the MAME object environment and return the result
local function execute_lua(code)
    local func, err = load(code, nil, "t", _G)
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

local function per_frame()
    local command = sock:read(1024)  
    if #command > 0 then
        emu.print_debug("Received command: " .. command)
        local result = execute_lua(command)
        sock:write(result .. "\n")  -- Add newline as a "message end" marker 
    end
end

emu.register_frame_done(per_frame)

emu.add_machine_stop_notifier(function() sock:close() end)



-- e.g. 
-- mame joust -window -nomouse -nokeys -skip_gameinfo -sound none -lua_plugin console_wrapper
-- ./mamed -window -debug -skip_gameinfo -autoboot_script ~/Dropbox/code/mamegym/console_wrapper.lua joust
