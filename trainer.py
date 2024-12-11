from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Correct import
import gymnasium
from gymnasium import spaces

from buddy import StockTradingEnv
import numpy as np

# Load preprocessed NumPy data
asts_data = np.load('./data/asts_stock_data.npy')

# Create the environment
env = StockTradingEnv(asts_data)

# Print out the action space info for debugging
print("Action Space:", env.action_space)
print("Type of Action Space:", type(env.action_space))

# Wrap environment in DummyVecEnv for stable baselines compatibility (even if you're using 1 environment)
new_env = DummyVecEnv([lambda: env])


# Initialize PPO agent
model = PPO("MlpPolicy", new_env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("./models/ppo_stock_trading_v1")
