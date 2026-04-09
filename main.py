import numpy as np
from framework import ExperimentFramework

def main():
    framework = ExperimentFramework(
        env_size=5,
        base_episodes=1000,
        switch_episode=500,
        num_trials=10,
        output_dir="results"
    )

    # Define parameter ranges and sample counts
    parameters = {
        'learning_rate': (0.01, 0.9),
        'discount_factor': (0.01, 0.99),
        'epsilon': (0.01, 0.9),
    }
    num_values = 15

    for param_name, param_range in parameters.items():
        print(f"\nProcessing parameter: {param_name}")

        # Generate log-spaced values across parameter range
        log_values = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), num=num_values)

        print(f"Testing {len(log_values)} values for {param_name}:")
        print(np.round(log_values, 4))

        print(f"Running baseline phase for {param_name}...")
        baseline = framework.run_baseline(param_name, log_values)

        print(f"\nRunning adaptation phase for {param_name}...")
        adaptation = framework.run_adaptation(param_name, log_values)

        framework.analyse_results(param_name, baseline, adaptation)

if __name__ == "__main__":
    main()