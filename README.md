# Ready Player Zero 

This project provides an [OpenAI Gym](https://gymnasium.farama.org/)-compatible environment for MAME (Multiple Arcade Machine Emulator) games, initially Joust. It allows reinforcement learning agents to interact with the game through a Python interface.

## Files

1. `mame_gym.py`: Contains the main `JoustEnv` class that implements the Gymnasium interface.
2. `mame_server.lua`: A Lua script that runs on the MAME side to enable remote control and data extraction.
3. `mame_client.py`: A Python client that communicates with the MAME server.

## Setup

1. Ensure you have [MAME](https://www.mamedev.org/) installed on your system.
2. Place `mame_server.lua` in a location accessible to MAME.
3. Install the required Python packages: ``` pip install gymnasium numpy ```
4. Install this package with ```pip install -e .``` from the project root to make it available to your python script located elsewhere. 

## Example Usage

1. Start MAME with the Joust ROM and useful options: 
    ```
    mame joust -window -skip_gameinfo -console -pause_brightness 1.0 -autoboot_script /path/to/mame_server.lua
    ```

2. In your Python script, you can now use the environment:

    ```python
    from mame_gym import JoustEnv

    env = JoustEnv()
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            observation, info = env.reset()

    env.close()
    ```
## Features
- Implements a simple protocol for control of MAME over TCP/IP
- Supports pausing, unpausing, and stepping through frames
- Allows reading game memory for score and lives
- Provides RGB observations of the game screen
- Implements a basic reward function based on score and lives

## Notes
- The environment is currently set up for the game Joust, but can be adapted for other MAME-supported games.
- Ensure that the MAME server is running before attempting to connect with the Python environment.
- This is very much a work in progress. Let's call it v0.4. It's (probably) functional.

## Future Improvements
- Implement support for multiple MAME games
- Explore a more performant communication method between MAME and Python
- Try/enable a more sophisticated reward function
- Implement an actual [RL agent](https://stable-baselines.readthedocs.io/en/master/) to demonstrate usage

## Acknowledgements

- Credit to M.J. Murray's [MAMEToolkit](https://github.com/M-J-Murray/MAMEToolkit) for inspiration. 
- Goals here are compatibility with the latest unmodified [MAME](https://github.com/mamedev/mame) release and to support a standard [Gym](https://gymnasium.farama.org/) interface.
