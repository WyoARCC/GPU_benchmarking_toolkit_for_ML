# Author: Tyson Limato
# project: GPU Benchmarking
# Purpose: Seamless visualization of Non-GPU statistics
# Start Date: 6/28/2023
import csv
import matplotlib.pyplot as plt


def plot_all_columns(file_path, savepath):
    """
    Plot All Columns

    This function plots all the columns against Core Time in a single graph.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

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
    ax1.plot(core_time, cpu_utilization, color='blue', label='CPU Utilization')
    ax1.set_ylabel('CPU Utilization')

    # Plot Thread Count
    ax1.plot(core_time, thread_count, color='red', label='Thread Count')
    ax1.set_ylabel('Thread Count')

    # Create a second y-axis for RAM Utilization
    ax2 = ax1.twinx()

    # Plot RAM Utilization (%)
    ax2.plot(core_time, ram_utilization_percent, color='green', label='RAM Utilization (%)')
    ax2.set_ylabel('RAM Utilization (%)')

    # Set x-axis label and title
    ax1.set_xlabel('Core Time')
    ax1.set_title('System Utilization')

    # Show legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot as an image file
    plt.savefig(savepath)
    plt.close()


def plot_cpu_utilization(file_path, savepath):
    """
    Plot CPU Utilization

    This function plots the CPU Utilization column against Core Time.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

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
    plt.plot(core_time, cpu_utilization, color='blue')
    plt.xlabel('Core Time')
    plt.ylabel('CPU Utilization')
    plt.title('CPU Utilization')
    # Save the plot as an image file
    plt.savefig(savepath)
    plt.close()


def plot_thread_count(file_path, savepath):
    """
    Plot Thread Count

    This function plots the Thread Count column against Core Time.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

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
    plt.plot(core_time, thread_count, color='red')
    plt.xlabel('Core Time')
    plt.ylabel('Thread Count')
    plt.title('Thread Count')
    # Save the plot as an image file
    plt.savefig(savepath)
    plt.close()


def plot_ram_utilization_percent(file_path, savepath):
    """
    Plot RAM Utilization (%)

    This function plots the RAM Utilization (%) column against Core Time.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

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
    plt.plot(core_time, ram_utilization_percent, color='green')
    plt.xlabel('Core Time')
    plt.ylabel('RAM Utilization (%)')
    plt.title('RAM Utilization (%)')
    # Save the plot as an image file
    plt.savefig(savepath)
    plt.close()


def plot_ram_utilization_mb(file_path, savepath):
    """
    Plot RAM Utilization (MB)

    This function plots the RAM Utilization (MB) column against Core Time.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

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
    plt.plot(core_time, ram_utilization_mb, color='orange')
    plt.xlabel('Core Time')
    plt.ylabel('RAM Utilization (MB)')
    plt.title('RAM Utilization (MB)')
    # Save the plot as an image file
    plt.savefig(savepath)
    plt.close()


def plot_training_loss(file_path, save_path):
    """
    Plot Training Loss

    This function plots the Training Loss column against the batch number.

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
    plt.plot(batch_number, training_loss, color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()


def plot_epoch_time(file_path, save_path):
    """
    Plot Epoch Time

    This function plots the Epoch Time column against the batch number.

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
    epoch_time = []

    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            batch_number.append(int(row[1]))
            epoch_time.append(float(row[3]))
    # Close the CSV file
    csv_file.close()
    # Plotting
    plt.plot(batch_number, epoch_time, color='red')
    plt.xlabel('Batch Number')
    plt.ylabel('Epoch Time')
    plt.title('Epoch Time')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()


def plot_throughput(file_path, save_path):
    """
    Plot Throughput

    This function plots the Throughput (Seq/sec) column against the batch number.

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
    plt.plot(batch_number, throughput, color='green')
    plt.xlabel('Batch Number')
    plt.ylabel('Throughput (Seq/sec)')
    plt.title('Throughput (Seq/sec)')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()


def plot_disk_iops(file_path, save_path):
    """
    Plot Disk IOPS

    This function plots the Disk Read IOPS and Disk Write IOPS columns against the batch number on the same graph.

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
    plt.plot(batch_number, disk_read_iops, color='purple', label='Disk Read IOPS')
    plt.plot(batch_number, disk_write_iops, color='orange', label='Disk Write IOPS')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk IOPS')
    plt.title('Disk Read IOPS vs. Disk Write IOPS')
    plt.legend()

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()


def plot_disk_read_iops(file_path, save_path):
    """
    Plot Disk Read IOPS

    This function plots the Disk Read IOPS column against the batch number.

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
    plt.plot(batch_number, disk_read_iops, color='purple')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Read IOPS')
    plt.title('Disk Read IOPS')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()


def plot_disk_write_iops(file_path, save_path):
    """
    Plot Disk Write IOPS

    This function plots the Disk Write IOPS column against the batch number.

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
    # Close the CSV file
    csv_file.close()
    # Plotting
    plt.plot(batch_number, disk_write_iops, color='orange')
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Write IOPS')
    plt.title('Disk Write IOPS')

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()
