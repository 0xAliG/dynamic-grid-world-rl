class GridWorld:
    """
    A 2D discrete navigation environment where an agent moves to a goal.

    The world is represented as a square grid of size N x N. The agent starts 
    at the top-left (0, 0) and must navigate to the goal position to 
    receive a positive reward.
    """
    def __init__(self, size: int = 5, goal_pos: tuple[int, int] = (4, 4)):
        self.current_pos = None
        self.size = size
        self.goal_pos = goal_pos
        self.reset()
        
        # Action space: up (0), right (1), down (2), left (3)
        self.action_space = 4
        
    def reset(self) -> int:
        """Reset environment to starting state."""
        self.current_pos = (0, 0)  # Start at top-left corner
        return self._get_state()
        
    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Take action in environment.
        Returns: (next_state, reward, done)
        """
        # Calculate new position
        x, y = self.current_pos
        if action == 0:   # up
            y = max(0, y - 1)
        elif action == 1: # right
            x = min(self.size - 1, x + 1)
        elif action == 2: # down
            y = min(self.size - 1, y + 1)
        elif action == 3: # left
            x = max(0, x - 1)
            
        self.current_pos = (x, y)
        
        done = self.current_pos == self.goal_pos
        reward = 1.0 if done else -0.1 
        
        return self._get_state(), reward, done
    
    def _get_state(self) -> int:
        """Convert current position to state number."""
        x, y = self.current_pos
        return y * self.size + x