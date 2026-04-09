import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
from tqdm import tqdm
from dynamic_grid_world import DynamicGridWorld
from Q_learning import QLearning
from visualiser import Visualiser

class ExperimentFramework:
    """
    ExperimentFramework handles the orchestration of training trials, 
    environment switches, and data collection. It automates the testing 
    of how different hyperparameters affect agent 'recovery' after 
    environmental changes.
    """
    def __init__(self, env_size=5, base_episodes=1000, switch_episode=500,
                 num_trials=100, output_dir="results"):
        self.env_size = env_size
        self.base_episodes = base_episodes
        self.switch_episode = switch_episode
        self.num_trials = num_trials
        self.output_dir = output_dir
        self.state_space = env_size ** 2
        self.default_params = {'learning_rate': 0.1, 'discount_factor': 0.9, 'epsilon': 0.1}
        os.makedirs(output_dir, exist_ok=True)

    def create_agent(self, params):
        return QLearning(self.state_space, 4, **params)

    def run_trial(self, env, agent, visualiser=None, is_baseline=False):
        steps_history = []
        for episode in range(self.base_episodes):
            if not is_baseline and episode == self.switch_episode:
                env.switch_configuration()

            state, done, steps = env.reset(), False, 0
            while not done and steps < 100: # 100-step cap prevents infinite loops in unstable policies
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                if visualiser:
                    visualiser.record_step(episode, next_state)
                state, steps = next_state, steps + 1
            steps_history.append(steps)
        return steps_history

    def run_baseline(self, param_name, values):
        results = {}
        for value in tqdm(values):
            params = {**self.default_params, param_name: value}
            trials = [np.mean(self.run_trial(DynamicGridWorld(), self.create_agent(params), is_baseline=True)[-50:])
                      for _ in range(self.num_trials)]
            results[value] = {'mean': np.mean(trials), 'std': np.std(trials)}
        return results

    def run_adaptation(self, param_name, values):
        results = {}
        for value in tqdm(values):
            params = {**self.default_params, param_name: value}
            all_trials_steps = []

            for trial_idx in range(self.num_trials):
                env = DynamicGridWorld()
                agent = self.create_agent(params)
                visualiser = Visualiser(self.env_size) if trial_idx == 0 else None
                steps = self.run_trial(env, agent, visualiser)
                all_trials_steps.append(steps)

                if trial_idx == 0 and visualiser:
                    self._save_visualisations(visualiser, f"{param_name}_{value:.3g}")

            adaptations = [self._adaptation_time(t) for t in all_trials_steps]

            post_switch_perfs = [np.mean(t[self.switch_episode:self.switch_episode + 50]) for t in all_trials_steps]


            results[value] = {
                'episodes_needed_to_recover_mean': np.mean(adaptations),
                'episodes_needed_to_recover_std': np.std(adaptations),
                'post_switch_mean': np.mean(post_switch_perfs),
                'post_switch_std': np.std(post_switch_perfs)
            }

            formatted_value = f"{value:.3g}"
            self._save_curve(all_trials_steps, f"{param_name}_{formatted_value}")

        return results

    def _adaptation_time(self, steps):
        # Calculate performance baseline just before the environment switch
        baseline = np.mean(steps[self.switch_episode - 50:self.switch_episode])
        post_switch = steps[self.switch_episode:]
        window = 5
        
        # Identify how many episodes it takes for the moving average 
        # to return to pre-switch performance levels
        for i in range(len(post_switch) - window + 1):
            window_avg = np.mean(post_switch[i:i + window])
            if window_avg <= baseline:
                return i
        return len(post_switch)

    def _save_curve(self, trials, name, dpi=300):

        trials_array = np.array(trials)
        mean = np.mean(trials_array, axis=0)
        std = np.std(trials_array, axis=0)

        # Extract performance points to highlight recovery on the graph
        pre_min = (np.min(mean[self.switch_episode - 50:self.switch_episode]))
        post_min = (np.min(mean[self.switch_episode:]))

        plt.figure(figsize=(12, 7))
        plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

        plt.plot(mean, label='Mean steps', color='#0066cc')

        lower_bound = np.maximum(mean - std, 8)
        plt.fill_between(
            range(len(mean)),
            lower_bound,
            mean + std,
            alpha=0.2,
            color='#0066cc',
            label='±1 Std Dev'
        )

        plt.axvline(self.switch_episode, c='#cc0000', ls='--', lw=2,
                    label='Configuration switch')

        pre_min_idx = np.argmin(mean[self.switch_episode - 50:self.switch_episode]) + (self.switch_episode - 50)
        post_min_idx = np.argmin(mean[self.switch_episode:]) + self.switch_episode

        plt.scatter([pre_min_idx], [pre_min], s=100, c='green', label=f'Pre-switch min: {pre_min:.3g}')

        plt.scatter([post_min_idx], [post_min], s=100, c='red', label=f'Post-switch min: {post_min:.3g}')

        plt.grid(True, alpha=0.3)
        plt.title(f"Learning Curve - {name}", fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Steps to Goal', fontsize=12)

        actual_max = np.max(mean + std)
        y_min = max(0, np.min(mean - std) * 0.9)
        y_max = actual_max * 1.1
        plt.ylim(y_min, y_max)

        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/{name}_curve.png", dpi=dpi)
        plt.close()

    def _save_visualisations(self, visualiser, prefix):
        visualiser.plot_state_heatmaps(
            f"State Visitation - {prefix}",
            f"{self.output_dir}/{prefix}progression_heatmap.png"
        )
        visualiser.plot_cumulative_state_heatmap(
            f"Cumulative Visits - {prefix}",
            f"{self.output_dir}/{prefix}cumulative_heatmap.png"
        )

    def analyse_results(self, param_name, baseline, adaptation):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        values = sorted(baseline.keys())

        ax1.errorbar(values, [baseline[v]['mean'] for v in values],
                     yerr=[baseline[v]['std'] for v in values], fmt='-o')
        ax1.set_title(f"Baseline Performance vs {param_name}")
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Average Steps')
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))

        ax2.errorbar(values, [adaptation[v]['episodes_needed_to_recover_mean'] for v in values],
                     yerr=[adaptation[v]['episodes_needed_to_recover_std'] for v in values],
                     fmt='-o', color='orange')

        ax2.set_title(f"Adaptation Time vs {param_name}")
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Episodes to Recover')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{param_name}_analysis.png")
        plt.close()

        pd.DataFrame({
            param_name: [float(f"{v:.3g}") for v in values],
            'baseline_mean': [float(f"{baseline[v]['mean']:.3g}") for v in values],
            'episodes_needed_to_recover_mean': [float(f"{adaptation[v]['episodes_needed_to_recover_mean']:.3g}") for v
                                                in values],
            'episodes_needed_to_recover_std': [float(f"{adaptation[v]['episodes_needed_to_recover_std']:.3g}") for v in
                                               values],
            'post_switch_mean': [float(f"{adaptation[v]['post_switch_mean']:.3g}") for v in values],
            'post_switch_std': [float(f"{adaptation[v]['post_switch_std']:.3g}") for v in values]
        }).to_csv(f"{self.output_dir}/{param_name}_comparison.csv", index=False)