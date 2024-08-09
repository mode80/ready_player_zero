#!/bin/zsh

MAME_SCRIPT="${PWD}/scripts/run_mame.sh"

# Open a persistent terminal window and run the MAME script there
osascript <<EOF
tell application "Terminal"
    do script "zsh $MAME_SCRIPT"
    activate
end tell
EOF

# Wait for MAME to start (adjust the sleep time if needed)
sleep 6
