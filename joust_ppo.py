#%%
from joust_env import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env 


env = MinJoustEnv()
env.render_mode = "human"

# check_env(env)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

# save the tained model
model.save("ppo_joust_")

# model.load("ppo_joust_")

# Test the trained model
obs, _ = env.reset()
for _ in range(10_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

