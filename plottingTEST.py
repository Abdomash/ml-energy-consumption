import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Traverse Subdirectories
results_dir = 'results'
output_dir = 'plots'
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
            combined_df = pd.concat(dfs)
            combined_files[classifier_name][machine_name] = combined_df
    return combined_files

# Step 4: Generate Subplots for Each Classifier and Machine
def generate_subplots_for_classifiers_and_machines(combined_files):
    for classifier_name, machine_data in combined_files.items():
        for machine_name, df in machine_data.items():
            datasets = df['dataset_name'].unique()
            num_datasets = len(datasets)
            fig, axes = plt.subplots(num_datasets, 1, figsize=(10, 6*num_datasets))
            fig.suptitle(f'Classifier: {classifier_name} - Machine: {machine_name}', fontsize=16)
            for i, dataset in enumerate(datasets):
                dataset_df = df[df['dataset_name'] == dataset]
                axes[i].set_title(f'Dataset: {dataset}', fontsize=14)
                axes[i].set_xlabel('Energy Consumed', fontsize=12)
                axes[i].set_ylabel('Test Accuracy', fontsize=12)

                # Plot each parameter grid value as a separate scatter plot
                parameter_grid_values = dataset_df['param_grid'].unique()
                for param_grid_value in parameter_grid_values:
                    grid_results = dataset_df[dataset_df['param_grid'] == param_grid_value]
                    axes[i].scatter(grid_results['energy_consumed'], grid_results['test_accuracy'], label=param_grid_value)

                # Place the legend below the x-axis
                axes[i].legend(title='Param Grid', bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10, ncol=len(parameter_grid_values))
                
                # Manually adjust layout to prevent overlap
                fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

            # Save plot as file
            output_filename = f'{classifier_name}_{machine_name}_plot.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_subplots_for_classifiers_and_machines(combined_files)
