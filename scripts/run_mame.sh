#!/bin/zsh
cd ~/mame && \
./mame -window -skip_gameinfo -console -pause_brightness 1.0 -sound none -autoboot_script  /Users/user/Dropbox/code/ready_player_zero/ready_player_zero/mame_server.lua joust
