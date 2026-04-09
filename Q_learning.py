import numpy as np

class QLearning:
    """
    An implementation of the Q-Learning algorithm for reinforcement learning.
    
    This agent maintains a Q-table to estimate the value of state-action pairs
    and uses an epsilon-greedy policy for the exploration-exploitation tradeoff.
    """
    def __init__(self, state_space: int, action_space: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.q_table = np.zeros((state_space, action_space))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def choose_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        return int((np.argmax(self.q_table[state])))
    
    def learn(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update Q-table using Q-learning update rule."""
        self.q_table[state, action] = (1 - self.lr) * self.q_table[state, action] + self.lr * (reward + self.gamma * np.max(self.q_table[next_state]))