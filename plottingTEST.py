import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Step 1: Traverse Subdirectories to collect CSV files
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
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_files[classifier_name][machine_name] = combined_df
    return combined_files

# Step 3: Generate Combined Plots with Custom Legends for Classifiers
def generate_combined_plots_for_classifiers(combined_files):
    marker_styles = ['X', 'o']  # X for Machine 1, o for Machine 2
    marker_sizes = [100, 50]  # Larger size for Machine 1, standard for Machine 2
    # Add a specific color for grid search number 5 and adjust others as needed
    colors = ['red', 'green', 'blue', 'purple', 'orange']  # Include a specific color for grid 5
    
    for classifier_name, machine_data in combined_files.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Classifier: {classifier_name}', fontsize=16)
        machine_legend_elements = []
        grid_legend_elements = []
        used_grid_numbers = []

        # Enable grid with default settings
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        for machine_idx, (machine_name, df) in enumerate(machine_data.items()):
            marker = marker_styles[machine_idx % len(marker_styles)]
            size = marker_sizes[machine_idx % len(marker_sizes)]
            datasets = df['dataset_name'].unique()
            
            for dataset in datasets:
                dataset_df = df[df['dataset_name'] == dataset]
                
                for grid_search_number in dataset_df['grid_search_number'].unique():
                    color = colors[grid_search_number % len(colors)]
                    grid_results = dataset_df[dataset_df['grid_search_number'] == grid_search_number]
                    ax.scatter(grid_results['energy_consumed'], grid_results['test_accuracy'], marker=marker, s=size, color=color)
                    if grid_search_number not in used_grid_numbers:
                        grid_legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Grid {grid_search_number}', markerfacecolor=color, markersize=10))
                        used_grid_numbers.append(grid_search_number)
            
            machine_legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=f'Machine: {machine_idx + 1}', markerfacecolor='gray', markersize=10))

        ax.set_xlabel('Energy Consumed', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        
        # Create and place legends
        machine_legend = ax.legend(handles=machine_legend_elements, title='Machines', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.gca().add_artist(machine_legend)
        ax.legend(handles=grid_legend_elements, title='Grid Search Number', loc='upper left', bbox_to_anchor=(1.05, 0.5))
        
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.75, hspace=0.4, wspace=0.4)
        output_filename = f'{classifier_name}_combined_plot.png'
        output_path = os.path.join('plots', output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_combined_plots_for_classifiers(combined_files)
