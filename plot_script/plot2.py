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

# Read data from the first set of files
gpu_times1, cpu_times1 = read_execution_times('../kevin_results/artificial_testSizes/artificial_testSizes_csv.txt')

# Read data from the second set of files
gpu_times2, cpu_times2 = read_execution_times('../kevin_results/artificial_trainSizes/artificial_trainSizes_csv.txt')

# Plotting
plt.plot(range(1, len(gpu_times1)+1), gpu_times1, label='GPU Set 1')
plt.plot(range(1, len(cpu_times1)+1), cpu_times1, label='CPU Set 1')
plt.plot(range(1, len(gpu_times2)+1), gpu_times2, label='GPU Set 2')
plt.plot(range(1, len(cpu_times2)+1), cpu_times2, label='CPU Set 2')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('Execution Time (s)')
plt.title('Comparison of Execution Times (GPU vs CPU)')
plt.legend()

# Setting y-axis to logarithmic scale and enabling grid
plt.yscale('log')
plt.grid(True, which="both", linestyle='--')

# Set x-axis tick locations
max_len = max(len(gpu_times1), len(cpu_times1), len(gpu_times2), len(cpu_times2))
plt.xticks(range(1, max_len + 1))
'''
# Set y-axis ticks dynamically based on the data from all sets
all_times = gpu_times1 + cpu_times1 + gpu_times2 + cpu_times2
max_time = max(all_times)
min_time = min(all_times)
range_time = max_time - min_time
# Determine tick positions
ticks = []
for i in range(-6, 6):
    ticks.extend([10 ** i * j for j in range(1, 10)])
    ticks.extend([2 * 10 ** i * j for j in range(1, 10)])
ticks = [tick for tick in ticks if min_time <= tick <= max_time]
plt.yticks(ticks)
'''


# Save the plot in a higher resolution and in a different format 
plt.savefig('execution_times_plot2.png', dpi=300, bbox_inches='tight', format='png')

# Show plot
#plt.show()
