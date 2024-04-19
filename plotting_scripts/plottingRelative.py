import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Step 1: Traverse Subdirectories to collect CSV files
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plotting_results', 'RelativePlots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def collect_experiment_files(results_dir):
    experiment_files = {}
    for machine_name in os.listdir(results_dir):
        machine_path = os.path.join(results_dir, machine_name)
        if os.path.isdir(machine_path):
            for filename in os.listdir(machine_path):
                if filename.startswith('backup'):
                    continue
                classifier_name = filename.split('_')[1]
                experiment_files.setdefault(classifier_name, {}).setdefault(machine_name, []).append(os.path.join(machine_path, filename))
    return experiment_files

# Step 2: Combine Files by Classifier and Machine
def combine_files_by_classifier_and_machine(experiment_files):
    combined_files = {}
    for classifier_name, machine_data in experiment_files.items():
        combined_files[classifier_name] = {}
        for machine_name, files in machine_data.items():
            dfs = [pd.read_csv(file) for file in files]
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_files[classifier_name][machine_name] = combined_df
    return combined_files

# Step 3: Generate Improved Combined Plots with Relative Percentage Increases for Each Grid Search Number
def generate_improved_relative_percentage_plots(combined_files):
    for classifier_name, machine_data in combined_files.items():
        datasets = set()
        for df in machine_data.values():
            datasets.update(df['dataset_name'].unique())

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'{classifier_name.split("Classifier")[0]} - {dataset}', fontsize=16, ha='right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

            # Initialize lists for percentage increases
            percentage_increase_energy = []
            percentage_increase_accuracy = []
            grid_search_numbers = sorted(df['grid_search_number'].unique())

            baseline_energy = None
            baseline_accuracy = None

            # Calculate percentage increases relative to the baseline (first grid search number)
            for number in grid_search_numbers:
                current_energy = []
                current_accuracy = []
                for df in machine_data.values():
                    grid_df = df[(df['dataset_name'] == dataset) & (df['grid_search_number'] == number)]
                    if not grid_df.empty:
                        current_energy.append(grid_df['energy_consumed'].mean())
                        current_accuracy.append(grid_df['test_accuracy'].mean())

                if baseline_energy is None and current_energy:
                    baseline_energy = sum(current_energy) / len(current_energy)
                if baseline_accuracy is None and current_accuracy:
                    baseline_accuracy = sum(current_accuracy) / len(current_accuracy)

                # Only process further if baseline values have been established
                if baseline_energy is not None and baseline_accuracy is not None:
                    mean_energy = sum(current_energy) / len(current_energy) if current_energy else baseline_energy
                    mean_accuracy = sum(current_accuracy) / len(current_accuracy) if current_accuracy else baseline_accuracy
                    percentage_increase_energy.append(100 * (mean_energy - baseline_energy) / baseline_energy)
                    percentage_increase_accuracy.append(100 * (mean_accuracy - baseline_accuracy) / baseline_accuracy)

            # Plotting the percentage increases with improved visualization
            ax.plot(percentage_increase_energy, percentage_increase_accuracy, label='Relative Increase', marker='o', linestyle='-', color='red')

            # Use linear scale if the data range is small
            ax.set_xscale('linear')
            ax.set_yscale('linear')

            # Enhance y-axis tick labeling for better granularity
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}%'.format(y)))

            ax.set_xlabel('Relative % Increase in Energy Consumed', fontsize=12)
            ax.set_ylabel('Relative % Increase in Test Accuracy', fontsize=12)
            ax.legend(loc='best', title='Relative Performance Increase')

            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
            output_filename = f'{classifier_name}_{dataset}_improved_relative_percentage_plot.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_improved_relative_percentage_plots(combined_files)
