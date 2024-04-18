import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Traverse Subdirectories to collect CSV files
results_dir = 'results'
output_dir = 'plotsMean'
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

# Step 3: Generate a combined scatter plot for average percentage increases with logarithmic scale
def generate_average_scatter_plot(combined_files):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', 'D', '^', 'p']

    for classifier_idx, (classifier_name, machine_data) in enumerate(combined_files.items()):
        color = colors[classifier_idx % len(colors)]
        marker = markers[classifier_idx % len(markers)]
        
        average_energy_increase = []
        average_accuracy_increase = []
        grid_search_numbers = sorted(next(iter(machine_data.values()))['grid_search_number'].unique())

        for number in grid_search_numbers:
            percentage_energy_increases = []
            percentage_accuracy_increases = []

            for df in machine_data.values():
                baseline_df = df[df['grid_search_number'] == df['grid_search_number'].min()]
                grid_df = df[df['grid_search_number'] == number]
                
                for dataset in df['dataset_name'].unique():
                    baseline_data = baseline_df[baseline_df['dataset_name'] == dataset]
                    dataset_df = grid_df[grid_df['dataset_name'] == dataset]
                    
                    if baseline_data.empty or dataset_df.empty:
                        continue

                    percentage_increase_energy = 100 * (dataset_df['energy_consumed'].mean() - baseline_data['energy_consumed'].mean()) / baseline_data['energy_consumed'].mean()
                    percentage_increase_accuracy = 100 * (dataset_df['test_accuracy'].mean() - baseline_data['test_accuracy'].mean()) / baseline_data['test_accuracy'].mean()
                    
                    percentage_energy_increases.append(percentage_increase_energy)
                    percentage_accuracy_increases.append(percentage_increase_accuracy)

            # Calculate the average of percentage increases
            if percentage_energy_increases and percentage_accuracy_increases:
                average_energy_increase.append(np.mean(percentage_energy_increases))
                average_accuracy_increase.append(np.mean(percentage_accuracy_increases))
        
        # Plotting average percentage increases
        ax.scatter(average_energy_increase, average_accuracy_increase, label=classifier_name, color=color, marker=marker)

    ax.set_xlabel('% Average Increase in Energy Consumed', fontsize=12)
    ax.set_ylabel('% Average Increase in Test Accuracy', fontsize=12)
    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_yscale('log')
    ax.legend(loc='best', title='Classifiers')
    plt.title('Comparison of % Increase in Accuracy vs. % Increase in Energy for all Classifiers', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'average_data_scatter_plot.png'), bbox_inches='tight')
    plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_average_scatter_plot(combined_files)
