import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Step 1: Traverse Subdirectories to collect CSV files
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plotting_results', 'FullPlots')

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

# Step 3: Generate Combined Plots with Custom Legends for Classifiers
def generate_separate_line_plots_for_datasets(combined_files):
    machine_colors = ['blue', 'green']  # Color for each machine
    line_styles = ['-', '--']  # Solid line for Machine 1, dashed line for Machine 2
    
    for classifier_name, machine_data in combined_files.items():
        datasets = set()
        for df in machine_data.values():
            datasets.update(df['dataset_name'].unique())

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'{classifier_name.split("Classifier")[0]} - {dataset}', fontsize=16, ha='right')
            machine_legend_elements = []

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

            for machine_idx, (machine_name, df) in enumerate(machine_data.items()):
                color = machine_colors[machine_idx % len(machine_colors)]
                line_style = line_styles[machine_idx % len(line_styles)]
                dataset_df = df[df['dataset_name'] == dataset]
                sorted_df = dataset_df.sort_values(by='grid_search_number')

                ax.plot(sorted_df['energy_consumed'], sorted_df['test_accuracy'], label=f'Machine {machine_idx + 1}', marker='o', linestyle=line_style, color=color)
                machine_legend_elements.append(Line2D([0], [0], color=color, label=f'Machine {machine_idx + 1}', marker='o', linestyle=line_style))

            ax.set_xlabel('Energy Consumed (kWh)', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            
            # Create and place legend
            ax.legend(handles=machine_legend_elements, loc='best', title='Machines')
            
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.75, hspace=0.4, wspace=0.4)
            output_filename = f'{classifier_name}_{dataset}_line_plot.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

# Main execution
experiment_files = collect_experiment_files(results_dir)
combined_files = combine_files_by_classifier_and_machine(experiment_files)
generate_separate_line_plots_for_datasets(combined_files)

