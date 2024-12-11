import gymnasium
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gymnasium.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        
        # Action space: Buy (0), Sell (1), Hold (2)
        self.action_space = spaces.Discrete(3)

                # Observation space: Adjust based on your state representation
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(data.shape[1] + 2,),
            dtype=np.float32
        )
        
        # # Observation space: Stock data + current balance and position
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(data.shape[1] + 2,), dtype=np.float32
        # )
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, **kwargs):
        # Optionally set the seed for randomness
        if seed is not None:
            np.random.seed(seed)  # Example of setting seed for random number generation

        self.balance = self.initial_balance
        self.position = 0  # Shares held
        self.current_step = 0
        self.done = False
        self.total_value = self.balance
        return self._next_observation(), {}
    
    def _next_observation(self):
        obs = self.data[self.current_step]
        return np.append(obs, [self.balance, self.position])
    
    def step(self, action):
        current_price = self.data[self.current_step, 0]  # Assuming close price is first column
        reward = 0

        # Ensure current_price is non-zero before proceeding
        if current_price == 0:
            current_price = 1e-6  # Assign a small value to avoid division by zero or invalid operations

        if action == 0:  # Buy
            num_shares = self.balance // current_price
            self.position += num_shares
            self.balance -= num_shares * current_price
        elif action == 1:  # Sell
            self.balance += self.position * current_price
            self.position = 0
        elif action == 2:  # Hold
            pass
        
        # Update total portfolio value and calculate reward
        self.total_value = self.balance + self.position * current_price
        
        # Check for NaN values in total_value
        if np.isnan(self.total_value):
            self.total_value = self.balance  # Fallback in case of NaN

        reward = self.total_value - self.initial_balance

        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            print(reward)
            self.done = True
        
        # Truncated value (add logic to determine if the episode was truncated)
        truncated = False  # Example: set to True if you have a truncation condition (e.g., max number of steps)

        return self._next_observation(), reward, self.done, truncated, {}
