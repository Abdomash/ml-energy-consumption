import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = 'path_to_your_csv_file.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Calculate the cumulative energy consumed
df['cumulative_energy'] = df['energy_consumed'].cumsum()

# Set up the bar graph
fig, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = df['grid_search_number']

# Plot test accuracy
ax1.bar(index, df['test_accuracy'], bar_width, label='Test Accuracy', color='blue', alpha=0.6)
ax1.set_xlabel('Grid Search Number')
ax1.set_ylabel('Test Accuracy', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for cumulative energy
ax2 = ax1.twinx()
ax2.bar(index + bar_width, df['cumulative_energy'], bar_width, label='Cumulative Energy Consumed', color='red', alpha=0.6)
ax2.set_ylabel('Cumulative Energy Consumed (J)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Adjust the scale for energy to make it smaller
ax2.set_ylim([0, max(df['cumulative_energy']) * 1.5])

plt.title('Test Accuracy and Cumulative Energy Consumed per Grid Search Iteration')
fig.tight_layout()
plt.show()
