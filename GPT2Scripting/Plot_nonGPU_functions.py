# Author: Tyson Limato
# project: GPU Benchmarking
# Purpose: Seamless visualization of Non-GPU statistics
# Start Date: 6/28/2023
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm


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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
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
    ax1.scatter(core_time, thread_count, linestyle='-', marker='o', color='red', label='Thread Count', s=10)
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
    plt.savefig(save_path, dpi=300)
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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the header row
        # Assuming the first column is 'Core Time' and the second column is 'CPU Utilization'
        time_index = headers.index('Core Time')
        utilization_index = headers.index('CPU Utilization')

        for row in reader:
            core_time_value = row[time_index]
            cpu_utilization_value = row[utilization_index]

            # Skip any non-numeric values in the 'Core Time' column
            if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                core_time.append(float(core_time_value))
                cpu_utilization.append(float(cpu_utilization_value))
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
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_thread_count(file_path, save_path):
    core_time = []
    thread_count = []

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the header row
        # Assuming the first column is 'Core Time' and the third column is 'Thread Count'
        time_index = headers.index('Core Time')
        thread_count_index = headers.index('Thread Count')

        for row in reader:
            core_time_value = row[time_index]
            thread_count_value = row[thread_count_index]

            # Skip any non-numeric values in the 'Core Time' column
            if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                core_time.append(float(core_time_value))
                thread_count.append(int(thread_count_value))

    # Plotting
    plt.scatter(core_time, thread_count, linestyle='-', marker='o', color='red', label='Thread Count', s=10)
    plt.xlabel('Core Time')
    plt.ylabel('Thread Count')
    plt.title('Thread Count')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ram_utilization_percent(file_path, save_path):
    core_time = []
    ram_utilization_percent = []

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the header row
        # Assuming the first column is 'Core Time' and the fourth column is 'RAM Utilization (%)'
        time_index = headers.index('Core Time')
        ram_utilization_percent_index = headers.index('RAM Utilization (%)')

        for row in reader:
            core_time_value = row[time_index]
            ram_utilization_percent_value = row[ram_utilization_percent_index]

            # Skip any non-numeric values in the 'Core Time' column
            if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                core_time.append(float(core_time_value))
                ram_utilization_percent.append(float(ram_utilization_percent_value))
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
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ram_utilization_mb(file_path, save_path):
    core_time = []
    ram_utilization_mb = []

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the header row
        # Assuming the first column is 'Core Time' and the fifth column is 'RAM Utilization (MB)'
        time_index = headers.index('Core Time')
        ram_utilization_mb_index = headers.index('RAM Utilization (MB)')

        for row in reader:
            core_time_value = row[time_index]
            ram_utilization_mb_value = row[ram_utilization_mb_index]

            # Skip any non-numeric values in the 'Core Time' column
            if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                core_time.append(float(core_time_value))
                ram_utilization_mb.append(float(ram_utilization_mb_value))
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
    plt.savefig(save_path, dpi=300)
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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
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
    plt.savefig(save_path, dpi=300)
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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            throughput.append(float(row[4]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.scatter(batch_number, throughput, linestyle='-', marker='o', color='green', label='Throughput', s=10)
    plt.xlabel('Batch Number')
    plt.ylabel('Throughput (Seq/sec)')
    plt.title('Throughput (Seq/sec)')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(throughput, np.ones(window_size), 'valid') / window_size
    plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
                label='Moving Average (window = 5)', s=10)

    # Calculate and display the average value
    average_throughput = np.mean(throughput)
    plt.axhline(average_throughput, color='red', linestyle='--', label=f'Average Throughput: {average_throughput:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path, dpi=300)
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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_read_iops.append(float(row[5]))
            disk_write_iops.append(float(row[6]))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.scatter(batch_number, disk_read_iops, linestyle='-', marker='o', color='purple',
                label='Disk Read IOPS', s=10)
    plt.scatter(batch_number, disk_write_iops, linestyle='-', marker='o', color='orange',
                label='Disk Write IOPS', s=10)
    plt.xlabel('Batch Number')
    plt.ylabel('Disk IOPS')
    plt.title('Disk Read IOPS vs. Disk Write IOPS')
    plt.grid(True)

    # Calculate and plot the moving average for Disk Read IOPS
    window_size = 5
    moving_avg_read = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
    plt.scatter(batch_number[window_size - 1:], moving_avg_read, linestyle='-', color='blue',
                label='Moving Average (Read IOPS, window = 5)', s=10)

    # Calculate and plot the moving average for Disk Write IOPS
    moving_avg_write = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
    plt.scatter(batch_number[window_size - 1:], moving_avg_write, linestyle='-', color='green',
                label='Moving Average (Write IOPS, window = 5)', s=10)

    # Calculate and display the average values
    average_read_iops = np.mean(disk_read_iops)
    average_write_iops = np.mean(disk_write_iops)
    plt.axhline(average_read_iops, color='red', linestyle='--', label=f'Average Read IOPS: {average_read_iops:.2f}')
    plt.axhline(average_write_iops, color='purple', linestyle='--',
                label=f'Average Write IOPS: {average_write_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path, dpi=300)
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

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_read_iops.append(float(row[5].strip()))
    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.scatter(batch_number, disk_read_iops, linestyle='-', marker='o', color='purple',
                label='Disk Read IOPS', s=10)
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Read IOPS (I/O Operations per Second)')
    plt.title('Disk Read IOPS')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
    plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
                label='Moving Average (window = 5)', s=10)

    # Calculate and display the average value
    average_iops = np.mean(disk_read_iops)
    plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_write_iops(file_path, save_path):
    batch_number = []
    disk_write_iops = []

    with open(file_path, mode='r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            disk_write_iops.append(float(row[6].strip()))

    # Close the CSV file
    csv_file.close()

    # Plotting
    plt.scatter(batch_number, disk_write_iops, color='orange', label='Disk Write IOPS', s=10)
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Write IOPS')
    plt.title('Disk Write IOPS')
    plt.grid(True)

    # Calculate and plot the moving average
    window_size = 5
    moving_avg = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
    plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
                label='Moving Average (window = 5)', s=10)

    # Calculate and display the average value
    average_iops = np.mean(disk_write_iops)
    plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

    # Save the plot as an image file
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_log_file(log_file_path):
    """
    Parse the log file and extract GPU utilization data for each GPU.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.

    Returns
    -------
    dict
        A dictionary containing GPU utilization data for each GPU.
        The keys are GPU IDs, and the values are dictionaries containing lists of timestamps, power utilization,
        core utilization, and VRAM utilization for each GPU.
    """
    gpu_data = {}  # Dictionary to store GPU utilization data for each GPU

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    pbar = tqdm(total=total_lines, desc='Parsing GPU Log File', unit='line', dynamic_ncols=True)

    gpu_id = None
    timestamps = []
    power_utilization = []
    core_utilization = []
    vram_utilization = []

    for i, line in enumerate(lines):
        pbar.update(1)
        line = line.strip()
        if line.startswith('gpt2_deployment'):
            if gpu_id is not None:
                gpu_data[gpu_id] = {
                    'timestamps': timestamps,
                    'power_utilization': power_utilization,
                    'core_utilization': core_utilization,
                    'vram_utilization': vram_utilization,
                }
                timestamps = []
                power_utilization = []
                core_utilization = []
                vram_utilization = []

            gpu_id = int(line.split()[-1].split('_')[1])  # Get GPU ID
        elif '[' in line:
            timestamp_str = line.split()[1]
            power_str, core_str, vram_str = lines[i + 1].split('|')[1:4]
            timestamp = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
            power = int(power_str.split()[1])
            core = int(core_str.split()[1])
            vram = int(vram_str.split()[1])
            timestamps.append(timestamp)
            power_utilization.append(power)
            core_utilization.append(core)
            vram_utilization.append(vram)

    # Add the data for the last GPU
    if gpu_id is not None:
        gpu_data[gpu_id] = {
            'timestamps': timestamps,
            'power_utilization': power_utilization,
            'core_utilization': core_utilization,
            'vram_utilization': vram_utilization,
        }

    pbar.close()
    return gpu_data


def parse_gpu_info(gpu_info):
    """
    Parse GPU information from the log data.

    Parameters
    ----------
    gpu_info : list
        A list of lines containing GPU information.

    Returns
    -------
    dict
        A dictionary containing lists of timestamps, power utilization, core utilization,
        and VRAM utilization for a single GPU.

    """
    gpu_utilization = {
        'timestamps': [],
        'power_utilization': [],
        'core_utilization': [],
        'vram_utilization': []
    }

    for line in gpu_info:
        parts = line.split('|')
        timestamp = parts[0].strip()
        power_utilization = int(parts[1].split()[4])
        core_utilization = int(parts[2].split()[2])
        vram_utilization = int(parts[3].split()[1])

        gpu_utilization['timestamps'].append(timestamp)
        gpu_utilization['power_utilization'].append(power_utilization)
        gpu_utilization['core_utilization'].append(core_utilization)
        gpu_utilization['vram_utilization'].append(vram_utilization)

    return gpu_utilization


def plot_gpu_power(gpu_data, save_path):
    """
    Plot GPU Power Utilization for each GPU.

    Parameters
    ----------
    gpu_data : dict
        A dictionary containing GPU utilization data for each GPU.
        The keys are GPU IDs, and the values are dictionaries containing lists of timestamps and power utilization
        for each GPU.
    save_path : str
        The base path to save the image files.

    Returns
    -------
    None

    """
    for gpu_id, data in gpu_data.items():
        timestamps = data['timestamps']
        power_utilization = data['power_utilization']
        # Create a new figure for each GPU
        plt.figure()
        # Plot GPU Power Utilization with explicit label
        plt.plot(timestamps, power_utilization, label=f'GPU {gpu_id} Power', marker='o', linestyle='-',
                 markersize=3)

        plt.xlabel('Timestamp')
        plt.ylabel('Power Utilization (%)')
        plt.title('GPU Power Utilization')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file with the specified GPU ID
        plt.savefig(f'{save_path}_gpu_{gpu_id}_power_utilization.png', dpi=300)
        plt.close()


def plot_gpu_vram(gpu_data, save_path):
    """
    Plot VRAM Utilization for each GPU.

    Parameters
    ----------
    gpu_data : dict
        A dictionary containing GPU utilization data for each GPU.
        The keys are GPU IDs, and the values are dictionaries containing lists of timestamps and VRAM utilization
        for each GPU.
    save_path : str
        The base path to save the image files.

    Returns
    -------
    None

    """
    for gpu_id, data in gpu_data.items():
        timestamps = data['timestamps']
        vram_utilization = data['vram_utilization']
        # Create a new figure for each GPU
        plt.figure()
        # Plot VRAM Utilization with explicit label
        plt.plot(timestamps, vram_utilization, label=f'GPU {gpu_id} VRAM', marker='o', linestyle='-',
                 markersize=3)

        plt.xlabel('Timestamp')
        plt.ylabel('VRAM Utilization (%)')
        plt.title('GPU VRAM Utilization')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file with the specified GPU ID
        plt.savefig(f'{save_path}_gpu_{gpu_id}_vram_utilization.png', dpi=300)
        plt.close()


def plot_gpu_core(gpu_data, save_path):
    """
    Plot GPU Core Utilization for each GPU.

    Parameters
    ----------
    gpu_data : dict
        A dictionary containing GPU utilization data for each GPU.
        The keys are GPU IDs, and the values are dictionaries containing lists of timestamps and core utilization
        for each GPU.
    save_path : str
        The base path to save the image files.

    Returns
    -------
    None

    """
    for gpu_id, data in gpu_data.items():
        timestamps = data['timestamps']
        core_utilization = data['core_utilization']
        # Create a new figure for each GPU
        plt.figure()
        # Plot GPU Core Utilization with explicit label
        plt.plot(timestamps, core_utilization, label=f'GPU {gpu_id} Core Utilization', marker='o', linestyle='-',
                 markersize=3)

        plt.xlabel('Timestamp')
        plt.ylabel('Core Utilization (%)')
        plt.title('GPU Core Utilization')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file with the specified GPU ID
        plt.savefig(f'{save_path}_gpu_{gpu_id}_core_utilization.png', dpi=300)
        plt.close()


# PREPROCESSING FUNCTIONS INCASE OF OVER THE AIR TRANSFER ERRORS WITH KUBECTL

def remove_nul_from_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
        data = csv_file.read()

    # Remove NUL characters from the data
    clean_data = data.replace('\x00', '')

    # Write the clean data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        out_file.write(clean_data)


def remove_non_ascii(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
        data = csv_file.read()

    # Remove non-ASCII characters from the data
    clean_data = ''.join(char for char in data if ord(char) < 128)

    # Write the clean data to a new CSV file with proper encoding
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        out_file.write(clean_data)


def remove_redundant_headers(file_path):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # Find the index of the first occurrence of the header
    header_index = None
    for i, row in enumerate(data):
        if 'Epoch' in row[0] and 'Core' in row[1]:  # Assuming 'Epoch' and 'Core' are in the first two columns
            header_index = i
            break

    if header_index is not None:
        # Remove the redundant headers
        data = data[:header_index] + data[header_index + 1:]

        # Write the updated data back to the file
        with open(file_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(data)


def clean_strings_quotes_from_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        data = file.read()

    # Split the data by newline characters to get individual rows
    rows = data.strip().split('\n')

    # Remove double-quote characters from both ends of each value in each row
    cleaned_rows = []
    for row in rows:
        cleaned_row = ','.join(value.strip('"') for value in row.split(','))
        cleaned_rows.append(cleaned_row)

    # Join the cleaned rows back into a single string with newline characters
    cleaned_data = '\n'.join(cleaned_rows)

    # Write the cleaned data back to the file
    with open(file_path, mode='w', newline='') as file:
        file.write(cleaned_data)


def remove_quotes_from_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        for row in data:
            cleaned_row = [value.strip('"') for value in row]
            csv_writer.writerow(cleaned_row)


def clean_csv(input_file, output_file="default"):
    if output_file == "default":
        output_file = input_file

    # Read the CSV file
    with open(input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
        data = csv_file.read()

    # Remove NUL characters from the data
    clean_data = data.replace('\x00', '')

    # Write the clean data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        out_file.write(clean_data)

    # Read the cleaned CSV data
    cleaned_data = []
    with open(output_file, 'r', newline='', encoding='utf-8', errors='replace') as cleaned_csv_file:
        for line in cleaned_csv_file:
            cleaned_data.append(line.strip().split(','))

    if cleaned_data[-1] == ['']:
        cleaned_data.pop()
    # Find duplicate values in the "Batch" column
    header = cleaned_data[0]
    batch_index = header.index("Batch")
    unique_batches = set()
    final_cleaned_data = [header]
    for row in cleaned_data[1:]:
        if len(row) > batch_index:
            batch_value = row[batch_index]
            if batch_value not in unique_batches:
                unique_batches.add(batch_value)
                final_cleaned_data.append(row)

    # Write the final cleaned data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as final_out_file:
        for row in final_cleaned_data:
            final_out_file.write(','.join(row) + '\n')


# Function to measure execution time
def measure_time(func):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def preprocess_data():
    print("Preprocessing Data...")
    remove_redundant_headers('training_results.csv')
    remove_redundant_headers('CPU_RAM_Utilization.csv')
    remove_nul_from_csv('training_results.csv', 'training_results.csv')
    remove_nul_from_csv('CPU_RAM_Utilization.csv', 'CPU_RAM_Utilization.csv')

    remove_non_ascii('training_results.csv', 'training_results.csv')
    remove_non_ascii('CPU_RAM_Utilization.csv', 'CPU_RAM_Utilization.csv')
    clean_csv('training_results.csv')
    clean_strings_quotes_from_csv('training_results.csv')
    clean_strings_quotes_from_csv('CPU_RAM_Utilization.csv')

    remove_quotes_from_csv('training_results.csv')
    remove_quotes_from_csv('CPU_RAM_Utilization.csv')
    print("Finished Preprocessing.")


def generate_graphs():
    print("Generating Graphs...")
    plot_training_loss('training_results.csv', 'Graphs/training_loss.png')
    plot_throughput('training_results.csv', 'Graphs/throughput.png')
    plot_disk_read_iops('training_results.csv', 'Graphs/disk_read_iops.png')
    plot_disk_write_iops('training_results.csv', 'Graphs/disk_write_iops.png')
    plot_disk_iops('training_results.csv', 'Graphs/disk_iops.png')
    plot_cpu_utilization('CPU_RAM_Utilization.csv', 'Graphs/cpu_utilization_percent.png')
    plot_thread_count('CPU_RAM_Utilization.csv', 'Graphs/thread_count.png')
    plot_ram_utilization_percent('CPU_RAM_Utilization.csv', 'Graphs/ram_utilization_percent.png')
    plot_ram_utilization_mb('CPU_RAM_Utilization.csv', 'Graphs/ram_utilization_mb.png')


def preprocess_and_generate_graphs():
    preprocess_data()
    generate_graphs()


# Measure the time it takes to preprocess and generate graphs
measure_time(preprocess_and_generate_graphs)
