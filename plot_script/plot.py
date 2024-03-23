import matplotlib.pyplot as plt
import csv

# Read the CSV file
with open('../kevin_results/artificial_features/artificial_features_csv.txt', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    gpu_times = next(reader)
    cpu_times = next(reader)

# Trim any empty strings
gpu_times = [time.strip() for time in gpu_times if time.strip()]
cpu_times = [time.strip() for time in cpu_times if time.strip()]

# Convert times to floats
gpu_times = [float(time) for time in gpu_times]
cpu_times = [float(time) for time in cpu_times]


# Plotting
plt.plot(range(1, len(gpu_times)+1), gpu_times, label='GPU')
plt.plot(range(1, len(cpu_times)+1), cpu_times, label='CPU')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('Execution Time (s)')
plt.title('Comparison of Execution Times (GPU vs CPU)')
plt.legend()

# Setting y-axis to logarithmic scale and enabling grid
plt.yscale('log')
plt.grid(True, which="both", linestyle='--')

# Set x-axis tick locations
plt.xticks(range(1, len(gpu_times)+1))

# Save the plot
#plt.savefig('execution_times_plot1.png')
# Save the plot in a higher resolution and in a different format (e.g., PNG or PDF)
plt.savefig('execution_times_plot1.png', dpi=300, bbox_inches='tight', format='png')

# Show plot
#plt.show()