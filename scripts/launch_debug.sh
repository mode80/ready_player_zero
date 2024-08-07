#!/bin/zsh

# TMP_SCRIPT="/Users/user/sand/mame0268_/launch_mame.sh"

# UGLY BUT WORKS
# Create a temporary script file
TMP_SCRIPT=$(mktemp)

# Write the MAME launch command to the temporary script
cat << 'EOF' > $TMP_SCRIPT
#!/bin/zsh
cd /Users/user/sand/mame0268_ && \
./mamed -window -skip_gameinfo -sound none -autoboot_script /Users/user/Dropbox/code/mamegym/console_wrapper.lua joust
# echo "MAME process has exited. Press any key to close this window."
# read -k1 -s
EOF

# Make the temporary script executable
chmod +x $TMP_SCRIPT

# Open a new Terminal window and run the temporary script
osascript <<EOF
tell application "Terminal"
    do script "zsh $TMP_SCRIPT"
    activate
end tell
EOF

# Wait for MAME to start (adjust the sleep time if needed)
sleep 5

echo "MAME should now be running. Starting debug session..."

# Clean up the temporary script
rm $TMP_SCRIPT