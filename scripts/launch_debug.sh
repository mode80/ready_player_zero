#!/bin/zsh

MAME_SCRIPT="${PWD}/scripts/launch_mame.sh"

# Open a new Terminal window and run the MAME script
osascript <<EOF
tell application "Terminal"
    do script "zsh $MAME_SCRIPT"
    activate
end tell
EOF

# Wait for MAME to start (adjust the sleep time if needed)
sleep 5

echo "MAME should now be running. Starting debug session..."