import pandas as pd
import matplotlib.pyplot as plt

# Sample data 
# Format: (GPU, Kernel/API, Time)
data = [
    (0, 'kernel1', 40),
    (0, 'kernel2', 30),
    (0, 'cudaMemcpy', 20),
    (0, 'api', 10),
    (1, 'kernel1', 30),
    (1, 'kernel2', 35),
    (1, 'cudaMemcpy', 25),
    (1, 'api', 10),
]

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=['GPU', 'Operation', 'Time'])

# Calculate total time per GPU
total_time_per_gpu = df.groupby('GPU')['Time'].sum()

# Calculate percentage time per operation per GPU
df['Percentage'] = df.groupby('GPU')['Time'].apply(lambda x: (x / x.sum()) * 100).reset_index(drop=True)

# Define colors for each operation
color_map = {
    'kernel1': 'skyblue',
    'kernel2': 'salmon',
    'cudaMemcpy': 'lightgreen',
    'api': 'gold',
}

# Plot histograms
fig, axs = plt.subplots(len(total_time_per_gpu), 1, figsize=(10, 6))
if len(total_time_per_gpu) == 1:
    axs = [axs]  # Handle single GPU case
for gpu, ax in zip(total_time_per_gpu.index, axs):
    gpu_data = df[df['GPU'] == gpu]
    ax.bar(gpu_data['Operation'], gpu_data['Percentage'], color=[color_map[op] for op in gpu_data['Operation']])
    ax.set_title(f'GPU {gpu}')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)  # Set y-axis limit to percentage scale
    ax.set_yticks(range(0, 101, 10))  # Set y-axis ticks from 0 to 100 with intervals of 10
    ax.grid(True, linestyle=':', linewidth=0.5, color='lightgrey')  # Add a light grid

plt.tight_layout()  # Adjust layout to prevent overlap

# Save the plot in a higher resolution and in a different format 
plt.savefig('histo_execution_times_plot.jpg', dpi=300, bbox_inches='tight', format='jpg')

