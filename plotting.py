import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_graph(csv_file_path, subdirectory_name):
    df = pd.read_csv(csv_file_path)
    df['cumulative_energy'] = df['energy_consumed'].cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = df['grid_search_number']

    bars1 = ax1.bar(index, df['test_accuracy'], bar_width, label='Test Accuracy', color='blue', alpha=0.6)
    ax1.set_xlabel('Grid Search Number')
    ax1.set_ylabel('Test Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(index + bar_width, df['cumulative_energy'], bar_width, label='Cumulative Energy Consumed', color='red', alpha=0.6)
    ax2.set_ylabel('Cumulative Energy Consumed (J)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, max(df['cumulative_energy']) * 1.5])

    # Annotate bars with the elapsed time
    for bar, elapsed_time in zip(bars2, df['elapsed_time']):
        ax2.annotate(f'{elapsed_time:.2f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.title(f'{subdirectory_name} - Test Accuracy and Cumulative Energy Consumed')

    # Ensure the plots directory exists
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot in the plots directory
    plot_filename = f"{subdirectory_name}_{os.path.basename(csv_file_path).replace('.csv', '')}.png"
    plt.savefig(os.path.join(plots_dir, plot_filename))
    plt.close(fig)

def process_directory(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and not file.startswith('backup_'):
                csv_file_path = os.path.join(subdir, file)
                subdirectory_name = os.path.basename(subdir)
                generate_graph(csv_file_path, subdirectory_name)

if __name__ == "__main__":
    process_directory('results')
