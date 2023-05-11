import matplotlib.pyplot as plt
import matplotlib.ticker
import sys
import argparse
import datetime


class cpu_usage_event:
    """	
    Object that stores the CPU utilization information for a particular point in time.	
    """
    def __init__(self, time_stamp, threads, cpu, vmsize, vmrss, rchar, wchar):
        self.time_stamp = time_stamp
        self.threads = float(threads)
        self.cpu = float(cpu)
        self.vmsize = float(vmsize)
        self.vmrss = float(vmrss)
        self.rchar = float(rchar)
        self.wchar = float(wchar)

class gpu_usage_event:
    """	
    Object that stores the information from a line in GPU stat.	
    """
    def __init__(self, node, time_stamp, driver, gpu_type, temp, fan_usage, utilization, encoding, decoding,
                 power_use, power_max, ram_use, ram_max):
        self.node = node
        self.time_stamp = time_stamp
        self.driver = driver
        self.gpu_type = gpu_type
        self.temp = temp
        self.fan_usage = fan_usage
        self.utilization = utilization
        self.encoding = encoding
        self.decoding = decoding
        self.power_use = power_use
        self.power_max = power_max
        self.ram_use = ram_use
        self.ram_max = ram_max


def get_month_number(month_in):
    """	
    Reads an abbreviated string representation of a month and returns an int representation of the month.	
    :param month_in: String, Three letter representation of the month.	
    :return: Int, Number representation of the month.	
    """
    if month_in == "Jan":
        return 1
    elif month_in == "Feb":
        return 2
    elif month_in == "Mar":
        return 3
    elif month_in == "Apr":
        return 4
    elif month_in == "May":
        return 5
    elif month_in == "Jun":
        return 6
    elif month_in == "Jul":
        return 7
    elif month_in == "Aug":
        return 8
    elif month_in == "Sep":
        return 9
    elif month_in == "Oct":
        return 10
    elif month_in == "Nov":
        return 11
    elif month_in == "Dec":
        return 12
    else:
        return 0


def process_gpu_log(logfile):
    """	
    Takes in an open dump of gpustat constructs a list of gpu_usage_events.	
    :param logfile: Open text file.	
    :return: List of gpu_usage_events.	
    """
    events = []
    header = None
    usage = None
    for line in logfile:
        if line[0] == "[":
            usage = line
            usage = usage.strip()
            usage_list = usage.split("|")
        else:
            header = line
            header = header.strip()
            header_list = header.split()
            continue
        if header is None or usage is None:
            continue
        else:
            time_raw = header_list[4].split(":")
            time_event = datetime.datetime(int(header_list[5]), get_month_number(header_list[2]),
                                                         int(header_list[3]), int(time_raw[0]), int(time_raw[1]),
                                                         int(time_raw[2]))
            utilization = usage_list[1].split()
            ram_usage = usage_list[2].split()
            events.append(
                gpu_usage_event(header_list[0], time_event, header_list[6], usage_list[0], utilization[0],
                                utilization[1], float(utilization[3]), float(utilization[6]), float(utilization[9]),
                                float(utilization[11]), float(utilization[13]), float(ram_usage[0]),
                                float(ram_usage[2])))
    return events


def process_cpu_log(logfile):
    """	
    Reads the files produced by memprof and constructs a list of cpu_usage_events.	
    :param logfile: List of memprof log files.	
    :return: List of cpu_usage_events.	
    """
    events = []
    # Skip the header in the CSV file.
    for i in range(6):
        next(logfile)
    for log_line in logfile:
        usage = log_line.split(',')
        # Convert the Linux epoch to datetime object.
        time_stamp = datetime.datetime.fromtimestamp(int(usage[0]))
        threads = usage[1]
        cpu = usage[2]
        vmsize = usage[3]
        vmrss = usage[4]
        rchar = usage[5]
        wchar = usage[6]
        events.append(cpu_usage_event(time_stamp, threads, cpu, vmsize, vmrss, rchar, wchar))
    return events


def create_gpu_plot(events, keys, height, width, event_interval, no_legend, save_name, plot_title, x_label, y_label,
                    source, cpu_events, sub_plot_flag):
    """
    Creates a plot from a dictionary of GPU events with a list of GPUs that the events correspond to.
    :param events: Dictionary of GPU events.
    :param keys: List of GPUs that the events correspond to.
    :param height: The height of the generated plot.
    :param width: The width of the generated plot.
    :param event_interval: The interval spacing in the x-axis in minutes.
    :param no_legend: Boolean flag for displaying or not displaying the legend for the plot.
    :param save_name: Base name to save plot to.
    :param plot_title: Title for the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param source:  What to generate the plot for. (GPU, POWER, or VRAM) Will default to VRAM.
    :param cpu_events: List of CPU utilization events.
        :param sub_plot_flag: Boolean used to determine if utilization polt is to be split into subplots.
        """
    if source == "GPU":
        s = 1
    elif source == "POWER":
        s = 2
    else:
        s = 3
    if sub_plot_flag is True:
        fig, axs = plt.subplots(2, figsize=(width, height))
    else:
        fig, axs = plt.subplots(figsize=(width, height))
    if isinstance(keys, list):
        for item in keys:
            if sub_plot_flag is True:
                axs[0].plot(events[item][0], events[item][s], label=item)
                axs[0].set_title(f'GPU {plot_title}')
            else:
                axs.plot(events[item][0], events[item][s], label=item)
    else:
        if sub_plot_flag is True:
            axs[0].plot(events[keys][0], events[keys][s], label=keys)
            axs[0].set_title(f'GPU {plot_title}')
        else:
            axs.plot(events[keys][0], events[keys][s], label=keys)
    if cpu_events is not None and source == "GPU":
        cpu_timestamps = []
        cpu_utilization = []
        for i in range(len(cpu_events)):
            cpu_timestamps.append(cpu_events[i].time_stamp)
            cpu_utilization.append(cpu_events[i].cpu)
        if sub_plot_flag is True:
            axs[1].plot(cpu_timestamps, cpu_utilization, label='CPU Utilization')
            axs[1].set_title(f'CPU {plot_title}')
        else:
            axs.plot(cpu_timestamps, cpu_utilization, label='CPU Utilization')
    if sub_plot_flag is True:
        for ax in axs:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
            ax.xaxis.get_ticklocs(minor=True)
            ax.set(xlabel=x_label, ylabel=y_label)
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=45, horizontalalignment='right')
            if no_legend is False:
                ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    else:
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(plot_title)
        axs.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
        axs.xaxis.get_ticklocs(minor=True)
        for label in axs.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')
        if no_legend is False:
            axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(f'{save_name}_{source}_Utilization.png')


def create_cpu_plot(cpu_events, height, width, event_interval, no_legend, save_name):
    """
    Creates individual plots for the information pulled from the memplot files.
    :param cpu_events: List of cpu events.
    :param height: The height of the generated plot.
    :param width: The width of the generated plot.
    :param event_interval: The interval spacing in the x-axis in minutes.
    :param no_legend: Boolean flag for displaying or not displaying the legend for the plot.
    :param save_name: Base name to save plot to.
    """
    fig, axs = plt.subplots(figsize=(width, height))
    cpu_timestamps = []
    threads = []
    cpu_utilization = []
    vmsize = []
    vmrss = []
    rchar = []
    wchar = []
    for i in range(len(cpu_events)):
        cpu_timestamps.append(cpu_events[i].time_stamp)
        threads.append(cpu_events[i].threads)
        cpu_utilization.append(cpu_events[i].cpu)
        vmsize.append(cpu_events[i].vmsize)
        vmrss.append(cpu_events[i].vmrss)
        rchar.append(cpu_events[i].rchar)
        wchar.append(cpu_events[i].wchar)

    # Create CPU Utilization Plot
    ylabel = "Percent Utilization"
    xlabel = "Run Time (Minutes)"
    title = "CPU Utilization"
    axs.plot(cpu_timestamps, cpu_utilization, label="CPU Utilization")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    axs.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
    axs.xaxis.get_ticklocs(minor=True)
    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(f'{save_name}_CPU_Utilization.png')

    # Create Thread Utilization Plot
    fig, axs = plt.subplots(figsize=(width, height))
    ylabel = "Threads"
    title = "Thread Utilization"
    axs.plot(cpu_timestamps, threads, label="Thread Count")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    axs.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
    axs.xaxis.get_ticklocs(minor=True)
    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(f'{save_name}_THREAD_Utilization.png')

    # Create IO Utilization Plot
    fig, axs = plt.subplots(figsize=(width, height))
    ylabel = "I/O Utilization (Bytes/s)"
    title = "I/O Utilization"
    axs.plot(cpu_timestamps, rchar, label='Data Read')
    axs.plot(cpu_timestamps, wchar, label='Data Written')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    axs.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
    axs.xaxis.get_ticklocs(minor=True)
    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(f'{save_name}_IO_Utilization.png')

    # Create RAM Utilization Plot
    fig, axs = plt.subplots(figsize=(width, height))
    ylabel = "RAM Utilization (kB)"
    title = "RAM Utilization"
    axs.plot(cpu_timestamps, vmsize, label='VmSize')
    axs.plot(cpu_timestamps, vmrss, label='VmRSS')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    axs.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(event_interval))
    axs.xaxis.get_ticklocs(minor=True)
    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(f'{save_name}_RAM_Utilization.png')


def shift_gpu_event_times(gpu_events, keys, zero_time):
    """
    Shifts the timestamps of the GPU events to be centered around a zero time.
    :param gpu_events: Dictionary of GPU event objects.
    :param keys: List of GPUs that the events correspond to.
    :param zero_time: Timestamp for the time to use as the new zero of the plot.
    """
    if isinstance(keys, list):
        for key in keys:
            for i in range(len(gpu_events[key][0])):
                gpu_events[key][0][i] = (gpu_events[key][0][i].timestamp() - zero_time) / 60
    else:
        for i in range(len(gpu_events[keys][0])):
            gpu_events[keys][0][i] = (gpu_events[keys][0][i].timestamp() - zero_time) / 60


def shift_cpu_event_times(cpu_events, zero_time):
    """
    Shifts the timestamps of the CPU events to be centered around a zero time.
    :param cpu_events: List of CPU event objects.
    :param zero_time: Timestamp for the time to use as the new zero of the plot.
    """
    for i in range(len(cpu_events)):
        cpu_events[i].time_stamp = (cpu_events[i].time_stamp.timestamp() - zero_time) / 60


def main():
    cl_parser = argparse.ArgumentParser(
        prog='gpustat_plot',
        description='Reads a file containing the output of the gpustat program that has been running in watch mode '
                    'and generates GPU utilization plots for each unique process.'
    )
    cl_parser.add_argument('filename', help='File to process.')
    cl_parser.add_argument('-H', '--height', type=float, default=5, help='Height for generated plots. '
                                                                         'Default is 5 inches.')
    cl_parser.add_argument('-W', '--width', type=float, default=10, help='Width for generated plots. '
                                                                         'Default is 10 inches.')
    cl_parser.add_argument('-i', '--interval', type=int, default=30, help='Interval in minutes between labels on x axis.'
                                                                          ' Default is 30.')
    cl_parser.add_argument('-n', '--no-legend', action='store_true', help='Remove the legend for generated plots.')
    cl_parser.add_argument('-r', '--rename', type=str, help='Base file name to rename generated plots to.')
    cl_parser.add_argument('-s', '--split', action='store_true', help='Split multiple GPUs out into separate plots.')
    cl_parser.add_argument('-p', '--sub-plot', action='store_true', help='Display GPU and CPU utilization as subplots.')
    cl_parser.add_argument('-o', '--offset', type=int, default=0, help='Time in hours to offset the GPU timing.')
    cl_parser.add_argument('--cpu-files', nargs='+', help='List of memprof.csv files. '
                                                          'Ex) memprof-42.csv memprof-1138.csv...')
    cl_args = cl_parser.parse_args()
    logfile_name = cl_args.filename
    height = cl_args.height
    width = cl_args.width
    no_legend = cl_args.no_legend
    event_interval = cl_args.interval
    cpu_files = cl_args.cpu_files
    if cl_args.rename is not None:
        save_name = cl_args.rename
    else:
        save_name = cl_args.filename
    try:
        logfile_handle = open(logfile_name, 'r')
    except OSError:
        print(f"Could not open {logfile_name} please check the file path and permissions.")
        sys.exit(-1)
    events = process_gpu_log(logfile_handle)
    logfile_handle.close()
    # The GPU events are in chronological order but if multiple GPUs where used then that data will be mixed in.
    sorted_events = {}
    # Get current timestamp in seconds.
    zero_time = datetime.datetime.today().timestamp()
    for event in events:
        gpu_id = event.gpu_type
        if gpu_id not in sorted_events:
            sorted_events.update({gpu_id: [[], [], [], []]})
        sub_list = sorted_events[gpu_id]
        times = sub_list[0]
        offset_time = event.time_stamp + datetime.timedelta(hours=cl_args.offset)
        times.append(offset_time)
        if offset_time.timestamp() < zero_time:
            zero_time = offset_time.timestamp()
        gpu_utilization = sub_list[1]
        gpu_utilization.append(event.utilization)
        power_utilization = sub_list[2]
        power_utilization.append(event.power_use)
        vram_utilization = sub_list[3]
        vram_utilization.append(event.ram_use)
        sorted_events[gpu_id] = sub_list
    event_key_list = sorted(sorted_events)
    if cpu_files is not None:
        cpu_events = []
        for file in cpu_files:
            cpu_log_file = open(file, 'r')
            cpu_events = cpu_events + process_cpu_log(cpu_log_file)
            cpu_log_file.close()
        cpu_events = sorted(cpu_events, key=lambda x: x.time_stamp)
        if zero_time < cpu_events[0].time_stamp.timestamp():
            zero_time = cpu_events[0].time_stamp.timestamp()
        shift_cpu_event_times(cpu_events, zero_time)
    else:
        cpu_events = None
    shift_gpu_event_times(sorted_events, event_key_list, zero_time)
    if cl_args.split is True:
        for i in range(len(event_key_list)):
            file_name = f"{save_name}_{i}"
            # Plot percent GPU utilization.
            plot_name = f"{i}_GPU_Utilization"
            xlabel = "Run Time (Minutes)"
            ylabel = "Percent Utilization"
            source = "GPU"
            create_gpu_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                            plot_name, xlabel, ylabel, source, cpu_events, cl_args.sub_plot)
            # Plot power utilization.
            plot_name = f"{i}_Power Utilization"
            ylabel = f"GPU Power Utilization"
            source = "POWER"
            create_gpu_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                            plot_name, xlabel, ylabel, source, cpu_events, False)
            # Plot VRAM utilization.
            plot_name = f"{i}_VRAM Utilization"
            ylabel = f"VRAM Utilization"
            source = "VRAM"
            create_gpu_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                            plot_name, xlabel, ylabel, source, cpu_events, False)
    else:
        # Plot percent GPU utilization.
        plot_name = "GPU Utilization"
        xlabel = "Run Time (Minutes)"
        ylabel = "Percent Utilization"
        source = "GPU"
        create_gpu_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                        plot_name, xlabel, ylabel, source, cpu_events, cl_args.sub_plot)
        # Plot power utilization.
        plot_name = "GPU Power Utilization"
        ylabel = "GPU Power Utilization"
        source = "POWER"
        create_gpu_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                        plot_name, xlabel, ylabel, source, cpu_events, False)
        # Plot VRAM utilization.
        plot_name = "VRAM Utilization"
        ylabel = "VRAM Utilization"
        source = "VRAM"
        create_gpu_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                        plot_name, xlabel, ylabel, source, cpu_events, False)
    if cpu_files is not None:
        create_cpu_plot(cpu_events, height, width, event_interval, no_legend, save_name)


if __name__ == "__main__":
    main()
