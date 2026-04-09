from grid_world import GridWorld

class DynamicGridWorld(GridWorld):
    def __init__(self, size: int = 5,
                 configurations: list[tuple[tuple[int, int], list[tuple[int, int]]]] = None):
        """
        Initialise DynamicGridWorld with multiple configurations.
        
        Args:
            size: size of the grid
            configurations: list of configurations, each containing:
                - goal position (tuple)
                - list of obstacle positions (list of tuples)
        """
        self.configurations = configurations or [
            # Configuration 1: goal at bottom right
            ((4, 4), [(0, 2), (1, 2), (2, 2), (3, 2)]),
            # Configuration 2: same goal position, different obstacles
            ((4, 4), [(1, 2), (2, 2), (3, 2), (4, 2)]) 
        ]
        
        super().__init__(size=size, goal_pos=self.configurations[0][0])
        
        self.current_config_idx = 0
        self.obstacles = self.configurations[0][1]

    def switch_configuration(self) -> None:
        """Switch to the next environment configuration."""
        self.current_config_idx = (self.current_config_idx + 1) % len(self.configurations)
        new_config = self.configurations[self.current_config_idx]
        self.goal_pos = new_config[0]
        self.obstacles = new_config[1]

    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Returns: (next_state, reward, done)
        - Reward +1.0 for reaching goal
        - Reward -1.0 for hitting obstacle
        - Reward -0.1 otherwise
        """
        x, y = self.current_pos
        new_x, new_y = x, y  # Default to current position

        if action == 0:   # Up
            new_y = max(0, y - 1)
        elif action == 1: # Right
            new_x = min(self.size - 1, x + 1)
        elif action == 2: # Down
            new_y = min(self.size - 1, y + 1)
        elif action == 3: # Left
            new_x = max(0, x - 1)

        new_pos = (new_x, new_y)
        hit_obstacle = new_pos in self.obstacles
        reached_goal = new_pos == self.goal_pos

        # Update position if move is valid
        if not hit_obstacle:
            self.current_pos = new_pos

        if reached_goal:
            reward = 1.0
            done = True
        elif hit_obstacle:
            reward = -1.0
            done = False
        else:
            reward = -0.1
            done = False

        return self._get_state(), reward, done