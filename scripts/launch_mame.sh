#!/bin/zsh
cd /Users/user/sand/mame0268_ && \
./mamed -window -skip_gameinfo -sound none -autoboot_script /Users/user/Dropbox/code/mamegym/console_wrapper.lua joust
echo "MAME process has exited. Press any key to close this window."
read -k1 -s