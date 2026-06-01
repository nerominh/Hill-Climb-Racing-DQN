# import gymnasium as gym
# import hill_racing_env  # registers the environment

# env = gym.make("hill_racing_env/HillRacing-v0", render_mode="human")
# obs, info = env.reset(seed=42)

# for _ in range(2000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     if terminated or truncated:
#         obs, info = env.reset()

# env.close()

import gymnasium as gym
import hill_racing_env

# 1. Initialize the environment using the specific ID
env = gym.make("hill_racing_env/HillRacing-v0")

# 2. Reset the environment to start a new episode.
# Note: In current Gymnasium versions, reset() returns a tuple containing the observation and an info dictionary.
observation, info = env.reset()

# 3. Print the resulting observation
print("Sample Observation:")
print(observation)

# 4. Cleanly close the environment
env.close()