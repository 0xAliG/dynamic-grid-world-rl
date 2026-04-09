# Q-Learning Adaptation in Dynamic Grid Worlds

This experiment analyses how Q-Learning agents adapt to sudden changes in grid world environments. 
The environment switches obstacle configurations mid-training, testing the agent's ability to recover performance under different hyperparameters.

## Key Features
- **Dynamic grid world** with configurable obstacles/goals
- **Hyperparameter analysis** (learning rate, discount factor, exploration epsilon)
- **Performance metrics**: baseline efficiency, adaptation speed, post-switch recovery
- **Visualisations** of learning curves and state visitation patterns

## Project Structure
- `grid_world.py`: Basic grid environment
- `dynamic_grid_world.py`: Environment with switchable configurations
- `Q_learning.py`: Q-Learning agent implementation
- `framework.py`: Experiment management and analysis
- `main.py`: Entry point
- `visualiser.py`: Heatmap generation

# Setup & Usage Guide

## Requirements
- **Python 3.13+**
- Required packages:
  ```bash
  pip install numpy pandas matplotlib seaborn tqdm
  
## Running the Experiment
- Execute with default settings:
- ```bash
  python main.py
  ```
  
## Output
After running, check the `results/` folder for:
- `*_curve.png`: Learning curves showing adaptation
- `*_heatmap.png`: State visitation patterns
- `*_comparison.csv`: Numerical metrics
- `*_analysis.png`: Parameter performance comparisons