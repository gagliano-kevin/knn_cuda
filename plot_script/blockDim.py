import matplotlib.pyplot as plt
import csv

def read_execution_times(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        gpu_times = next(reader)

    # Trim any empty strings
    gpu_times = [time.strip() for time in gpu_times if time.strip()]

    # Convert times to floats
    gpu_times = [float(time) for time in gpu_times]

    return gpu_times

# Read data from the first set of files
gpu_times1 = read_execution_times('../kevin_results/artificial_blockDims/artificial_blockDims_csv.txt')

# Read data from the second set of files
gpu_times2 = read_execution_times('../gianeh_results/artificial_blockDims/artificial_blockDims_csv.txt')

# Plotting  
plt.plot([2, 4, 8, 16, 32], gpu_times1, label='GPU Setup 1')
plt.plot([2, 4, 8, 16, 32], gpu_times2, label='GPU Setup 2')

# Adding labels and title
plt.xlabel('Block Dimension')
plt.ylabel('Execution Time (s)')
plt.title('Execution Times for Test on Block Dimension')

plt.legend()

# Setting y-axis to logarithmic scale and enabling grid
plt.yscale('log')
plt.grid(True, which="both", linestyle='--')

# Set x-axis tick locations
plt.xticks([2, 4, 8, 16, 32])

# Save the plot in a higher resolution and in a different format 
plt.savefig('blockDim.jpg', dpi=300, bbox_inches='tight', format='jpg')


