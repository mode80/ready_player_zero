import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))# Add the parent directory to the Python path

from mame_gym.mame_gym import JoustEnv

import numpy as np

def test_joust_env():
    # Test initialization
    env = JoustEnv()
    assert env.mame is not None, "MAME client not initialized"
    assert env.observation_space.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation space"
    assert env.action_space.n == 4, "Incorrect action space"

    # Test reset
    observation, info = env.reset()
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape"
    assert env.last_score == {1: 0, 2: 0}, "Scores not reset"
    assert env.last_lives == {1: 3, 2: 3}, "Lives not reset"

    # Test step
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape after step"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(done, bool), "Done is not a boolean"
    assert isinstance(info, dict), "Info is not a dictionary"

    # Test render
    rendered = env.render()
    assert rendered is None or rendered.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect render output"

    # Test close
    env.close()
    # You might want to add an attribute to check if the environment is closed
    # assert env.is_closed, "Environment not properly closed"

    print("All tests passed!")

def test_helper_methods():
    env = JoustEnv()

    # Test _get_screen_size
    width, height = env._get_screen_size()
    assert width == JoustEnv.WIDTH and height == JoustEnv.HEIGHT, "Incorrect screen size"

    # Test _get_frame_number
    frame_number = env._get_frame_number()
    assert isinstance(frame_number, int), "Frame number is not an integer"

    # Test _get_pixels
    pixels = env._get_pixels()
    assert len(pixels) >= JoustEnv.WIDTH * JoustEnv.HEIGHT * 4, "Incorrect pixels data"

    # Test _read_byte and _read_word
    byte_value = env._read_byte(JoustEnv.P1_LIVES_ADDR)
    assert 0 <= byte_value <= 255, "Invalid byte value"
    word_value = env._read_word(JoustEnv.P1_SCORE_ADDR)
    assert 0 <= word_value <= 65535, "Invalid word value"

    # Test _get_lives and _get_score
    lives = env._get_lives(player=1)
    assert 0 <= lives <= 3, "Invalid number of lives"
    score = env._get_score(player=1)
    assert score >= 0, "Invalid score"

    # Test _calculate_reward
    reward = env._calculate_reward(player=1)
    assert -1.0 <= reward <= 1.0, "Reward out of expected range"

    # Test _check_done
    done = env._check_done(player=1)
    assert isinstance(done, bool), "Done is not a boolean"

    # Test _action_to_input
    action = env.action_space.sample()
    port, field = env._action_to_input(action)
    assert port in env.inputs, "Invalid port returned"
    assert field in env.inputs[port], "Invalid field returned"

    env.close()
    print("All helper method tests passed!")

if __name__ == "__main__":
    test_joust_env()
    test_helper_methods()
