import pandas as pd
import matplotlib.pyplot as plt

# Data from: /knn_cuda/kevin_results/nvprof_outputs/single_iteration_profiling/nvprof_iris_1_iter.txt
# Format: (GPU, Kernel/API call, %Time)
data = [
    (1, 'knnDistances (2.05%)', 2.05),
    (1, 'knn (95.21%)', 95.21),
    (1, 'memcpy HtoD (1.21%)', 1.21),
    (1, 'memcpy DtoH (0.33%)', 0.33),
    (1, 'memset (1.19%)', 1.19),
]

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=['GPU', 'Operation', 'Time'])

# Calculate total time per GPU
total_time_per_gpu = df.groupby('GPU')['Time'].sum()

# Calculate percentage time per operation 
df['Percentage'] = df.groupby('GPU')['Time'].apply(lambda x: (x / x.sum()) * 100).reset_index(drop=True)

# Define colors for each operation
color_map = {
    'knnDistances (2.05%)': '#4682B4',      # Steel Blue
    'knn (95.21%)': '#FF6347',              # Tomato
    'memcpy HtoD (1.21%)': '#32CD32',       # Lime Green
    'memcpy DtoH (0.33%)': '#4169E1',       # Royal Blue
    'memset (1.19%)': '#FFAA00',            # Orange
}

# Plot histograms
fig, axs = plt.subplots(len(total_time_per_gpu), 1, figsize=(10, 6))
if len(total_time_per_gpu) == 1:
    axs = [axs]                             # Handle single GPU case
for gpu, ax in zip(total_time_per_gpu.index, axs):
    gpu_data = df[df['GPU'] == gpu]
    ax.bar(gpu_data['Operation'], gpu_data['Percentage'], color=[color_map[op] for op in gpu_data['Operation']])
    ax.set_title(f'GPU {gpu}')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)                     # Set y-axis limit to percentage scale
    ax.set_yticks(range(0, 101, 10))        # Set y-axis ticks from 0 to 100 with intervals of 10
    ax.grid(True, linestyle=':', linewidth=0.5, color='lightgrey')  # Add a light grid

plt.tight_layout()                          # Adjust layout to prevent overlap

# Save the plot in a higher resolution and in a different format 
plt.savefig('histogram_iris.jpg', dpi=300, bbox_inches='tight', format='jpg')

