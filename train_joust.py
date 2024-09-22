import gym

env = JoustEnv(render_mode='human')
observation = env.reset()

for episode in range(10):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()