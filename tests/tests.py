import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ready_player_zero import JoustEnv
import numpy as np

def test_atomic_functions(env):
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

def test_action_and_observation(env):
    env.reset()

    # Test _queue_input
    env._queue_input(JoustEnv.LEFT)
    assert env._commands_are_processing(), "Command queue should not be empty after _queue_input"

    # Test _get_observation
    observation = env._get_observation()
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape"

    print("All action and observation tests passed!")

def test_game_logic(env):
    # Test _calculate_reward
    env.last_score = 0
    env.last_lives = 3
    reward = env._calculate_reward(score=100, lives=3)
    assert -1.0 <= reward <= 1.0, "Reward out of expected range"

    # Test _check_done
    done = env._check_done(score=100, lives=0)
    assert isinstance(done, bool), "Done is not a boolean"

    print("All game logic tests passed!")

def test_environment_lifecycle(env):
    # Test reset
    observation, info = env.reset()
    assert observation.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation shape"
    assert env.last_score == 0, "Score not reset"
    assert env.last_lives == 3, "Lives not reset"

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

def test_joust_env(env):
    assert env.client is not None, "MAME client not initialized"
    assert env.observation_space.shape == (JoustEnv.HEIGHT, JoustEnv.WIDTH, 3), "Incorrect observation space"
    assert env.action_space.n == 6, "Incorrect action space"
    assert len(env.actions) == 6, "Incorrect number of actions"
    print("JoustEnv initialization test passed!")

if __name__ == "__main__":
    env = JoustEnv()
    test_atomic_functions(env)
    test_action_and_observation(env)
    test_game_logic(env)
    test_environment_lifecycle(env)
    test_joust_env(env)
    env.close()
    print("All tests completed successfully!")
