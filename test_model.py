import gymnasium
from stable_baselines3 import PPO

# Assuming you have saved the model during training:
model = PPO.load("models/ppo_stock_trading_v1.zip")

# Create the environment again (use your specific env setup)
env = gymnasium.make('YourStockTradingEnv-v0')

# Reset environment
obs = env.reset()

# Run a test episode
done = False
total_reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()  # Optional: Render the environment if you want to visualize it

print(f"Total reward for the test episode: {total_reward}")
