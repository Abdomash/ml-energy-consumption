import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Step 1: Traverse Subdirectories to collect CSV files
results_dir = os.path.join('..', 'results')
output_dir = os.path.join('..', 'plotting_results', 'MeanPlots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def collect_experiment_files(results_dir):
    experiment_files = {}
    for machine_name in os.listdir(results_dir):
        machine_path = os.path.join(results_dir, machine_name)
        if os.path.isdir(machine_path):
            for filename in os.listdir(machine_path):
                if filename.startswith('backup_'):
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

# Step 3: Generate Combined Plots with Averages for Each Grid Search Number
def generate_average_plots(combined_files):
    for classifier_name, machine_data in combined_files.items():
        datasets = set()
        for df in machine_data.values():
            datasets.update(df['dataset_name'].unique())

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'{classifier_name.split("Classifier")[0]} - {dataset}', fontsize=16, ha='right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

            # Initialize lists for averaged values
            average_energy = []
            average_accuracy = []
            grid_search_numbers = sorted(df['grid_search_number'].unique())

            # Calculate mean values for each grid search number
            for number in grid_search_numbers:
                energy = []
                accuracy = []
                for df in machine_data.values():
                    grid_df = df[(df['dataset_name'] == dataset) & (df['grid_search_number'] == number)]
                    energy.append(grid_df['energy_consumed'].mean())
                    accuracy.append(grid_df['test_accuracy'].mean())

                average_energy.append(sum(energy) / len(energy))
                average_accuracy.append(sum(accuracy) / len(accuracy))

            # Plotting the average values
            ax.plot(average_energy, average_accuracy, label='Average', marker='o', linestyle='-', color='red')

            ax.set_xlabel('Average Energy Consumed', fontsize=12)
            ax.set_ylabel('Average Test Accuracy', fontsize=12)
            ax.legend(loc='best', title='Mean Performance')

            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.75, hspace=0.4, wspace=0.4)
            output_filename = f'{classifier_name}_{dataset}_average_plot.png'
            output_path = os.path.join('plotsMean', output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_average_plots(combined_files)
