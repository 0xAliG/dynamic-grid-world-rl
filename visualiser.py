import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualiser:
    """
    Handles data collection and heatmap generation for agent state visitation patterns.
    
    This class tracks how often an agent visits each coordinate in a grid world,
    organising data into 'chunks' of episodes to show how exploration changes over time.
    """
    def __init__(self, env_size: int, chunk_size: int = 100):
        self.env_size = env_size
        self.chunk_size = chunk_size
        self.trajectory_history = []
        self.state_visits_by_chunk = []

    def record_step(self, episode: int, state: int) -> None:
        chunk_index = episode // self.chunk_size
        while len(self.state_visits_by_chunk) <= chunk_index:
            self.state_visits_by_chunk.append(np.zeros((self.env_size, self.env_size)))

        y = state // self.env_size
        x = state % self.env_size
        self.state_visits_by_chunk[chunk_index][y, x] += 1

    def plot_state_heatmaps(self, title_prefix: str = "State Visitation Frequency", save_path: str = None):
        """Plot and save heatmaps for each episode chunk."""
        n_chunks = len(self.state_visits_by_chunk)
        cols = min(3, n_chunks)
        rows = (n_chunks + cols - 1) // cols

        plt.figure(figsize=(5 * cols, 4 * rows))
        plt.suptitle(f"{title_prefix} Progression", fontsize=16)

        for chunk_idx, visits in enumerate(self.state_visits_by_chunk):
            plt.subplot(rows, cols, chunk_idx + 1)
            sns.heatmap(visits, annot=True, fmt='.0f', cmap='YlOrRd', cbar=False, square=True)
            plt.title(f"Episodes {chunk_idx * self.chunk_size}-{(chunk_idx + 1) * self.chunk_size}")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_cumulative_state_heatmap(self, title: str = "Cumulative State Visitation", save_path: str = None):
        """Plot and save cumulative heatmap."""
        cumulative = np.sum(self.state_visits_by_chunk, axis=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cumulative, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()