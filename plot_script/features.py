import matplotlib.pyplot as plt
import csv

def read_execution_times(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        gpu_times = next(reader)
        cpu_times = next(reader)

    # Trim any empty strings
    gpu_times = [time.strip() for time in gpu_times if time.strip()]
    cpu_times = [time.strip() for time in cpu_times if time.strip()]

    # Convert times to floats
    gpu_times = [float(time) for time in gpu_times]
    cpu_times = [float(time) for time in cpu_times]

    return gpu_times, cpu_times

# Read data from the first file
gpu_times1, cpu_times1 = read_execution_times('../kevin_results/artificial_features/artificial_features_csv.txt')

# Read data from the second file
gpu_times2, cpu_times2 = read_execution_times('../gianeh_results/artificial_features/artificial_features_csv.txt')

# Plotting  
plt.plot(range(10, 101, 10), gpu_times1, label='GPU Setup 1')
plt.plot(range(10, 101, 10), cpu_times1, label='CPU Setup 1')
plt.plot(range(10, 101, 10), gpu_times2, label='GPU Setup 2')
plt.plot(range(10, 101, 10), cpu_times2, label='CPU Setup 2')

# Adding labels and title
plt.xlabel('Number of Features')
plt.ylabel('Execution Time (s)')
plt.title('Execution Times for Test on Number of Features')

plt.legend()

# Setting y-axis to logarithmic scale and enabling grid
plt.yscale('log')
plt.grid(True, which="both", linestyle='--')

# Set x-axis tick locations
plt.xticks(range(10, 101, 10))

# Save the plot in a higher resolution and in a different format 
plt.savefig('features.jpg', dpi=300, bbox_inches='tight', format='jpg')


