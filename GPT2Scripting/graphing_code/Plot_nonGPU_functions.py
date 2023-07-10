# Author: Tyson Limato
# project: GPU Benchmarking
# Purpose: Seamless visualization of Non-GPU statistics
# Start Date: 6/28/2023
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_all_columns(file_path, save_path):
    """
    Plot All Columns

    This function plots all the columns against Core Time in a single graph,
    and provides a data summary.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    core_time = []
    cpu_utilization = []
    thread_count = []
    ram_utilization_percent = []
    ram_utilization_mb = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            core_time.append(float(row[0]))
            cpu_utilization.append(float(row[1]))
            thread_count.append(int(row[2]))
            ram_utilization_percent.append(float(row[3]))
            ram_utilization_mb.append(float(row[4]))

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot CPU Utilization
    ax1.plot(core_time, cpu_utilization, linestyle='-', marker='o', markersize=2, color='blue', label='CPU Utilization')
    ax1.set_ylabel('CPU Utilization')

    # Plot Thread Count
    ax1.plot(core_time, thread_count, linestyle='-', marker='o', markersize=2, color='red', label='Thread Count')
    ax1.set_ylabel('Thread Count')

    # Create a second y-axis for RAM Utilization
    ax2 = ax1.twinx()

    # Plot RAM Utilization (%)
    ax2.plot(core_time, ram_utilization_percent, linestyle='-', marker='o', markersize=2, color='green',
             label='RAMUtilization (%)')
    ax2.set_ylabel('RAM Utilization (%)')

    # Plot RAM Utilization (MB)
    ax2.plot(core_time, ram_utilization_mb, linestyle='-', marker='o', markersize=2, color='orange',
             label='RAM Utilization (MB)')
    ax2.set_ylabel('RAM Utilization (MB)')

    # Set x-axis label and title
    ax1.set_xlabel('Core Time')
    ax1.set_title('System Utilization')

    # Show legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()

    # Data Summary
    cpu_avg = sum(cpu_utilization) / len(cpu_utilization)
    thread_avg = sum(thread_count) / len(thread_count)
    ram_percent_avg = sum(ram_utilization_percent) / len(ram_utilization_percent)
    ram_mb_avg = sum(ram_utilization_mb) / len(ram_utilization_mb)

    print('Data Summary:')
    print(f'Average CPU Utilization: {cpu_avg:.2f}')
    print(f'Average Thread Count: {thread_avg:.2f}')
    print(f'Average RAM Utilization (%): {ram_percent_avg:.2f}')
    print(f'Average RAM Utilization (MB): {ram_mb_avg:.2f}')


def plot_cpu_utilization(file_path, save_path):
    """
    Plot CPU Utilization

    This function plots the CPU Utilization column against Core Time,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    core_time = []
    cpu_utilization = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            core_time.append(float(row[0]))
            cpu_utilization.append(float(row[1]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(core_time, cpu_utilization, linestyle='-', marker='o', markersize=2, color='blue', label='CPU Utilization')
    plt.xlabel('Core Time')
    plt.ylabel('CPU Utilization')
    plt.title('CPU Utilization')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(cpu_utilization, np.ones(window_size), 'valid') / window_size
    plt.plot(core_time[window_size - 1:], moving_avg, linestyle='-', color='green', label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_utilization = np.mean(cpu_utilization)
    plt.axhline(average_utilization, color='red', linestyle='--',
                label=f'Average Utilization: {average_utilization:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_thread_count(file_path, save_path):
    """
    Plot Thread Count

    This function plots the Thread Count column against Core Time,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    core_time = []
    thread_count = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            core_time.append(float(row[0]))
            thread_count.append(int(row[2]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(core_time, thread_count, linestyle='-', marker='o', markersize=2, color='red', label='Thread Count')
    plt.xlabel('Core Time')
    plt.ylabel('Thread Count')
    plt.title('Thread Count')
    plt.grid(True)

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_ram_utilization_percent(file_path, save_path):
    """
    Plot RAM Utilization (%)

    This function plots the RAM Utilization (%) column against Core Time,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    core_time = []
    ram_utilization_percent = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            core_time.append(float(row[0]))
            ram_utilization_percent.append(float(row[3]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(core_time, ram_utilization_percent, linestyle='-',
             marker='o', markersize=2, color='green', label='RAM Utilization (%)')
    plt.xlabel('Core Time')
    plt.ylabel('RAM Utilization (%)')
    plt.title('RAM Utilization (%)')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(ram_utilization_percent, np.ones(window_size), 'valid') / window_size
    plt.plot(core_time[window_size - 1:], moving_avg, linestyle='-', color='blue', label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_utilization = np.mean(ram_utilization_percent)
    plt.axhline(average_utilization, color='red', linestyle='--',
                label=f'Average Utilization: {average_utilization:.2f}%')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_ram_utilization_mb(file_path, save_path):
    """
    Plot RAM Utilization (MB)

    This function plots the RAM Utilization (MB) column against Core Time,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    core_time = []
    ram_utilization_mb = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            core_time.append(float(row[0]))
            ram_utilization_mb.append(float(row[4]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(core_time, ram_utilization_mb, linestyle='-',
             marker='o', markersize=2, color='orange', label='RAM Utilization (MB)')
    plt.xlabel('Core Time')
    plt.ylabel('RAM Utilization (MB)')
    plt.title('RAM Utilization (MB)')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(ram_utilization_mb, np.ones(window_size), 'valid') / window_size
    plt.plot(core_time[window_size - 1:], moving_avg, linestyle='-', color='blue', label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_utilization = np.mean(ram_utilization_mb)
    plt.axhline(average_utilization, color='red', linestyle='--',
                label=f'Average Utilization: {average_utilization:.2f} MB')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_training_loss(file_path, save_path):
    """
    Plot Training Loss

    This function plots the Training Loss column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = []
    training_loss = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            training_loss.append(float(row[2]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(batch_number, training_loss, linestyle='-', marker='o', markersize=3, color='blue', label='Training Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(training_loss, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg, linestyle='-',
             color='green', label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_loss = np.mean(training_loss)
    plt.axhline(average_loss, color='red', linestyle='--', label=f'Average Loss: {average_loss:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_throughput(file_path, save_path):
    """
    Plot Throughput

    This function plots the Throughput (Seq/sec) column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = []
    throughput = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            throughput.append(float(row[4]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(batch_number, throughput, linestyle='-', marker='o', markersize=3, color='green', label='Throughput')
    plt.xlabel('Batch Number')
    plt.ylabel('Throughput (Seq/sec)')
    plt.title('Throughput (Seq/sec)')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(throughput, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
             label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_throughput = np.mean(throughput)
    plt.axhline(average_throughput, color='red', linestyle='--', label=f'Average Throughput: {average_throughput:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_disk_iops(file_path, save_path):
    """
    Plot Disk IOPS

    This function plots the Disk Read IOPS and Disk Write IOPS columns against the batch number on the same graph,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = []
    disk_read_iops = []
    disk_write_iops = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_read_iops.append(float(row[5]))
            disk_write_iops.append(float(row[6]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(batch_number, disk_read_iops, linestyle='-', marker='o', markersize=3, color='purple',
             label='Disk Read IOPS')
    plt.plot(batch_number, disk_write_iops, linestyle='-', marker='o', markersize=3, color='orange',
             label='Disk Write IOPS')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk IOPS')
    plt.title('Disk Read IOPS vs. Disk Write IOPS')
    plt.grid(True)

    # Calculate and plot the moving average for Disk Read IOPS
    window_size = 5
    moving_avg_read = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg_read, linestyle='-', color='blue',
             label='Moving Average (Read IOPS, window = 5)')

    # Calculate and plot the moving average for Disk Write IOPS
    moving_avg_write = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg_write, linestyle='-', color='green',
             label='Moving Average (Write IOPS, window = 5)')

    # Calculate and display the average values
    average_read_iops = np.mean(disk_read_iops)
    average_write_iops = np.mean(disk_write_iops)
    plt.axhline(average_read_iops, color='red', linestyle='--', label=f'Average Read IOPS: {average_read_iops:.2f}')
    plt.axhline(average_write_iops, color='purple', linestyle='--',
                label=f'Average Write IOPS: {average_write_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_disk_read_iops(file_path, save_path):
    """
    Plot Disk Read IOPS

    This function plots the Disk Read IOPS column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = []
    disk_read_iops = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_read_iops.append(float(row[5]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(batch_number, disk_read_iops, linestyle='-', marker='o', markersize=3, color='purple',
             label='Disk Read IOPS')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Read IOPS (I/O Operations per Second)')
    plt.title('Disk Read IOPS')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
             label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_iops = np.mean(disk_read_iops)
    plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_disk_write_iops(file_path, save_path):
    """
    Plot Disk Write IOPS

    This function plots the Disk Write IOPS column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = []
    disk_write_iops = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_write_iops.append(float(row[6]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.plot(batch_number, disk_write_iops, linestyle='-', marker='o', markersize=3, color='orange',
             label='Disk Write IOPS')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Write IOPS')
    plt.title('Disk Write IOPS')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
    plt.plot(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
             label='Moving Average (window = 5)')

    # Calculate and display the average value
    average_iops = np.mean(disk_write_iops)
    plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# Plot Training Loss and save the image
plot_training_loss('training_results.csv', 'training_loss.png')
# Plot Throughput and save the image
plot_throughput('training_results.csv', 'throughput.png')
# Plot Disk Read IOPS and save the image
plot_disk_read_iops('training_results.csv', 'disk_read_iops.png')
# Plot Disk Write IOPS and save the image
plot_disk_write_iops('training_results.csv', 'disk_write_iops.png')
# Plot Disk IOPS and save the image
plot_disk_iops('training_results.csv', 'Old Graphs/disk_iops.png')
# Plot CPU Utilization and save the image
plot_cpu_utilization('CPU_RAM_Utilization.csv', 'cpu_utilization_percent.png')
# Plot Thread Count and save the image
plot_thread_count('CPU_RAM_Utilization.csv', 'thread_count.png')
# Plot RAM Utilization and save the image
plot_ram_utilization_percent('CPU_RAM_Utilization.csv', 'ram_utilization_percent.png')
