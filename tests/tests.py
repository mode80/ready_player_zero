import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import sleep

from ready_player_zero import JoustEnv
import numpy as np

def test_atomic_functions():

    # Test _read_byte and _read_dword
    byte_value = env._read_byte(JoustEnv.P1_LIVES_ADDR)
    assert 0 <= byte_value <= 255, "Invalid byte value"
    dword_value = env._read_dword(JoustEnv.P1_SCORE_ADDR)
    assert 0 <= dword_value <= 4294967295, "Invalid dword value"

    # Test _get_frame_number
    frame_number = env._get_frame_number()
    assert isinstance(frame_number, int), "Frame number is not an integer"

    # Test _get_pixels
    pixels = env._get_pixels()
    assert len(pixels) >= JoustEnv.HEIGHT * JoustEnv.WIDTH * 4, "Incorrect pixels data"

    # Test _get_screen_size
    width, height = env._get_screen_size()
    assert width == JoustEnv.WIDTH and height == JoustEnv.HEIGHT, "Incorrect screen size"

    # Test _get_lives and _get_score
    lives = env._get_lives()
    assert 0 <= lives <= 5, "Invalid number of lives"
    score = env._get_score()
    assert score >= 0, "Invalid score"

    # Test _commands_are_processing
    processing = env._commands_are_processing()
    assert isinstance(processing, bool), "Invalid return type for _commands_are_processing"

    # Test _get_lua_errors
    try:
        env._get_lua_errors()
    except Exception as e:
        assert False, f"_get_lua_errors raised an unexpected exception: {e}"

    print("All atomic function tests passed!")

def test_action_and_observation():

    env.reset() 

    # Test _send_input
    env._send_input(JoustEnv.LEFT)
    assert not env._commands_are_processing(), "Command queue should be empty after _send_input"

    # Test _step_frame
    env._step_frame()
    assert not env._commands_are_processing(), "Command queue should be empty after _step_frame"

    # Test _get_observation
    observation = env._get_observation()
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape"

    print("All action and observation tests passed!")

def test_game_logic():

    # Test _calculate_reward
    env.last_score = {1: 0, 2: 0}
    env.last_lives = {1: 3, 2: 3}
    reward = env._calculate_reward()
    assert -1.0 <= reward <= 1.0, "Reward out of expected range"

    # Test _check_done
    done = env._check_done()
    assert isinstance(done, bool), "Done is not a boolean"

    print("All game logic tests passed!")

def test_environment_lifecycle():

    # Test reset
    observation, info = env.reset()
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape"
    assert env.last_score == {1: 0, 2: 0}, "Scores not reset"
    assert env.last_lives[JoustEnv.PLAYER] == 3, "Lives not reset"

    # Test step
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape after step"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(done, bool), "Done is not a boolean"
    assert isinstance(info, dict), "Info is not a dictionary"

    # Test render
    rendered = env.render()
    assert rendered is None, "Render should return None when render_mode is None"

    print("All environment lifecycle tests passed!")

def test_joust_env():
    assert env.client is not None, "MAME client not initialized"
    assert env.observation_space.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation space"
    assert env.action_space.n == 4, "Incorrect action space"
    print("JoustEnv initialization test passed!")

env = JoustEnv()
test_atomic_functions()
test_action_and_observation()
test_game_logic()
test_environment_lifecycle()
test_joust_env()
env.close()
print("All tests completed successfully!")

