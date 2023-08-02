import matplotlib.pyplot as plt
import matplotlib.ticker
import sys
import argparse
import datetime
import os


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
            time_event = datetime.datetime(int(header_list[5]), get_month_number(header_list[2]), int(header_list[3]),
                                           int(time_raw[0]), int(time_raw[1]), int(time_raw[2]))
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


def create_plots_plot(events, keys, height, width, event_interval, no_legend, save_name, cpu_events, sub_plot_flag,
                      split_gpus, plot_cpu_utilization):
    """
    Creates the utilization plots for this dataset.
    :param events: Dictionary of list of GPU events.
    :param keys: List of GPUs that the events correspond to.
    :param height: The height of the generated plot.
    :param width: The width of the generated plot.
    :param event_interval: The interval spacing in the x-axis in minutes.
    :param no_legend: Boolean flag for displaying or not displaying the legend for the plot.
    :param save_name: Base name to save plot to.
    :param cpu_events: List of CPU utilization events.
    :param sub_plot_flag: Boolean used to determine if utilization plot is to be split into subplots.
    :param split_gpus: Boolean flag used to determine if the GPU events should be displayed on same plot or not.
    :param plot_cpu_utilization: Boolean flag used to determine if CPU utilization plot will be generated.
    """
    x_label = "Run Time (Minutes)"
    if cpu_events is not None:
        cpu_timestamps = []
        cpu_utilization = []
        threads = []
        vmsize = []
        vmrss = []
        rchar = []
        wchar = []
        for i in range(len(cpu_events)):
            cpu_timestamps.append(cpu_events[i].time_stamp)
            cpu_utilization.append(cpu_events[i].cpu)
            threads.append(cpu_events[i].threads)
            vmsize.append(cpu_events[i].vmsize)
            vmrss.append(cpu_events[i].vmrss)
            rchar.append(cpu_events[i].rchar)
            wchar.append(cpu_events[i].wchar)
        if plot_cpu_utilization is False:
            cpu_utilization = None
        y_label = "I/O Utilization (Bytes)"
        plot_title = "I/O Utilization"
        plot_item = "Data Read"
        sub_plot_item = "Data Written"
        create_plot(cpu_timestamps, rchar, plot_item, cpu_timestamps, wchar, sub_plot_item, height, width,
                    event_interval, no_legend, save_name, False, x_label, y_label, plot_title)
        y_label = "RAM Utilization (kB)"
        plot_title = "RAM Utilization"
        plot_item = "VmSize"
        sub_plot_item = "VmRSS"
        create_plot(cpu_timestamps, vmsize, plot_item, cpu_timestamps, vmrss, sub_plot_item, height, width,
                    event_interval, no_legend, save_name, False, x_label, y_label, plot_title)
        y_label = "Threads"
        plot_title = "Thread Utilization"
        plot_item = "Thread Count"
        create_plot(cpu_timestamps, threads, plot_item, None, None, None, height, width,
                    event_interval, no_legend, save_name, False, x_label, y_label, plot_title)
    else:
        cpu_timestamps = None
        cpu_utilization = None
    if split_gpus is True:
        index = 0
        for item in keys:
            save_to_this = f'{save_name}_{index}'
            gpu_timestamps = []
            gpu_utilization = []
            gpu_vram_utilization = []
            gpu_power_utilization = []
            for i in range(len(events[item])):
                gpu_timestamps.append(events[item][i].time_stamp)
                gpu_utilization.append(events[item][i].utilization)
                gpu_vram_utilization.append(events[item][i].ram_use)
                gpu_power_utilization.append(events[item][i].power_use)
            y_label = "Percent Utilization"
            plot_title = "Utilization"
            sub_plot_item = "CPU Utilization"
            create_plot(gpu_timestamps, gpu_utilization, item, cpu_timestamps, cpu_utilization, sub_plot_item, height,
                        width, event_interval, no_legend, save_to_this, sub_plot_flag, x_label, y_label, plot_title)
            y_label = "Power Used (Watts)"
            plot_title = "GPU Power Utilization"
            create_plot(gpu_timestamps, gpu_power_utilization, item, None, None, None, height,
                        width, event_interval, no_legend, save_to_this, sub_plot_flag, x_label, y_label, plot_title)
            y_label = "VRAM Utilization (MB)"
            plot_title = "VRAM Utilization"
            create_plot(gpu_timestamps, gpu_vram_utilization, item, None, None, None, height,
                        width, event_interval, no_legend, save_to_this, sub_plot_flag, x_label, y_label, plot_title)
            index = index + 1
    else:
        gpu_timestamp_dictionary = {}
        gpu_utilization_dictionary = {}
        gpu_vram_utilization_dictionary = {}
        gpu_power_utilization_dictionary = {}
        for item in keys:
            gpu_timestamps = []
            gpu_utilization = []
            gpu_vram_utilization = []
            gpu_power_utilization = []
            for i in range(len(events[item])):
                gpu_timestamps.append(events[item][i].time_stamp)
                gpu_utilization.append(events[item][i].utilization)
                gpu_vram_utilization.append(events[item][i].ram_use)
                gpu_power_utilization.append(events[item][i].power_use)
            gpu_timestamp_dictionary[item] = gpu_timestamps
            gpu_utilization_dictionary[item] = gpu_utilization
            gpu_vram_utilization_dictionary[item] = gpu_vram_utilization
            gpu_power_utilization_dictionary[item] = gpu_power_utilization
        y_label = "Percent Utilization"
        plot_title = "Utilization"
        sub_plot_item = "CPU Utilization"
        create_plot(gpu_timestamp_dictionary, gpu_utilization_dictionary, keys, cpu_timestamps, cpu_utilization,
                    sub_plot_item, height, width, event_interval, no_legend, save_name, sub_plot_flag, x_label, y_label, plot_title)
        y_label = "Power Used (Watts)"
        plot_title = "GPU Power Utilization"
        create_plot(gpu_timestamp_dictionary, gpu_power_utilization_dictionary, keys, None, None, None, height, width,
                    event_interval, no_legend, save_name, sub_plot_flag, x_label, y_label, plot_title)
        y_label = "VRAM Utilization (MB)"
        plot_title = "VRAM Utilization"
        create_plot(gpu_timestamp_dictionary, gpu_vram_utilization_dictionary, keys, None, None, None, height, width,
                    event_interval, no_legend, save_name, sub_plot_flag, x_label, y_label, plot_title)


def create_plot(timestamps, events, keys, subplot_timestamps, subplot_events, subplot_item, height, width,
                event_interval, no_legend, save_name, sub_plot_flag, x_label, y_label, plot_title):
    """
    Creates a plot with the timestamps on the X axis with the resource in events being on the Y axis.
    :param timestamps: List or Dictionary of Lists of timestamps.
    :param events: List or Dictionary of Lists of events to plot.
    :param keys: List of GPU names.
    :param subplot_timestamps: List of timestamps to plot or None.
    :param subplot_events: List of events to plot or None.
    :param subplot_item: Name of resource being plotted in subplot.
    :param height: The height of the generated plot.
    :param width: The width of the generated plot.
    :param event_interval: The interval spacing in the x-axis in minutes.
    :param no_legend: Boolean flag for displaying or not displaying the legend for the plot.
    :param save_name: Base name to save plot to.
    :param sub_plot_flag: Boolean used to determine if utilization plot is to be split into subplots.
    :param x_label: What to label X axis as.
    :param y_label: What to label Y axis as.
    :param plot_title: What to title the plot as.
    """
    if sub_plot_flag is True and subplot_events is not None:
        fig, axs = plt.subplots(2, figsize=(width, height))
    else:
        fig, axs = plt.subplots(figsize=(width, height))
    if isinstance(timestamps, dict):
        for i in range(len(keys)):
            if sub_plot_flag is True and subplot_events is not None:
                axs[0].plot(timestamps[keys[i]], events[keys[i]], label=keys[i])
            else:
                axs.plot(timestamps[keys[i]], events[keys[i]], label=keys[i])
    else:
        axs.plot(timestamps, events, label=keys)
    if subplot_events is not None and subplot_events is not None:
        if sub_plot_flag is True:
            axs[1].plot(subplot_timestamps, subplot_events, label=subplot_item)
        else:
            axs.plot(subplot_timestamps, subplot_events, label=subplot_item)
    if sub_plot_flag is True and subplot_events is not None:
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
    plt.savefig(f'{save_name}_{plot_title.replace(" ", "_").replace("/", "_")}.png')


def shift_gpu_event_times(gpu_events, zero_time):
    """
    Shifts the timestamps of the GPU events to be centered around a zero time.
    :param gpu_events: List of GPU event objects.
    :param zero_time: Timestamp for the time to use as the new zero of the plot.
    """
    for i in range(len(gpu_events)):
        gpu_events[i].time_stamp = (gpu_events[i].time_stamp.timestamp() - zero_time) / 60


def get_zero_time(gpu_events, cpu_events):
    """
    Gets the lowest timestamp from GPU and CPU events.
    :param gpu_events: List of GPU events objects.
    :param cpu_events: List of CPU events objects.
    :return: zero_time: Lowest timestamp from GPU and CPU events.
    """
    zero_time = datetime.datetime.today().timestamp()
    for i in range(len(gpu_events)):
        if gpu_events[i].time_stamp.timestamp() < zero_time:
            zero_time = gpu_events[i].time_stamp.timestamp()
    if cpu_events is not None:
        for i in range(len(cpu_events)):
            if cpu_events[i].time_stamp.timestamp() < zero_time:
                zero_time = cpu_events[i].time_stamp.timestamp()
    return zero_time


def shift_cpu_event_times(cpu_events, zero_time):
    """
    Shifts the timestamps of the CPU events to be centered around a zero time.
    :param cpu_events: List of CPU event objects.
    :param zero_time: Timestamp for the time to use as the new zero of the plot.
    """
    for i in range(len(cpu_events)):
        cpu_events[i].time_stamp = (cpu_events[i].time_stamp.timestamp() - zero_time) / 60


def print_statistics(gpu_events, keys, cpu_events):
    """
    Prints usage statistics to the terminal.
    :param gpu_events: Dictionary of list of GPU events.
    :param keys: The keys to gpu_events.
    :param cpu_events: List of CPU utilization events.
    """
    if cpu_events is not None:
        cpu_timestamps = []
        cpu_utilization = []
        threads = []
        vmsize = []
        vmrss = []
        rchar = []
        wchar = []

        for i in range(len(cpu_events)):
            cpu_timestamps.append(cpu_events[i].time_stamp)
            cpu_utilization.append(cpu_events[i].cpu)
            threads.append(cpu_events[i].threads)
            vmsize.append(cpu_events[i].vmsize)
            vmrss.append(cpu_events[i].vmrss)
            rchar.append(cpu_events[i].rchar)
            wchar.append(cpu_events[i].wchar)
        cpu_timestamp_min = min(cpu_timestamps)
        cpu_timestamp_max = max(cpu_timestamps)
        cpu_utilization_min = min(cpu_utilization)
        cpu_utilization_max = max(cpu_utilization)
        cpu_utilization_average = sum(cpu_utilization) / len(cpu_utilization)
        threads_min = min(threads)
        threads_max = max(threads)
        threads_average = sum(threads) / len(threads)
        vmsize_min = min(vmsize)
        vmsize_max = max(vmsize)
        vmsize_average = sum(vmsize) / len(vmsize)
        vmrss_min = min(vmrss)
        vmrss_max = max(vmrss)
        vmrss_average = sum(vmrss) / len(vmrss)
        rchar_min = min(rchar)
        rchar_max = max(rchar)
        rchar_average = sum(rchar) / len(rchar)
        wchar_min = min(wchar)
        wchar_max = max(wchar)
        wchar_average = sum(wchar) / len(wchar)
        print("-------------------------------------------------------------------------------------------------------")
        print("Summary Of Usage Of System Resources")
        print("-------------------------------------------------------------------------------------------------------")
        print(f"Time On CPU: {cpu_timestamp_max - cpu_timestamp_min} Minutes")
        print(f"Minimum CPU Utilization: {cpu_utilization_min}%")
        print(f"Maximum CPU Utilization: {cpu_utilization_max}%")
        print(f"Average CPU Utilization: {cpu_utilization_average}%")
        print(f"Minimum Number Of Threads: {threads_min}")
        print(f"Maximum Number Of Threads: {threads_max}")
        print(f"Average Number Of Threads: {threads_average}")
        print(f"Minimum VmSize: {vmsize_min} kB")
        print(f"Maximum VmSize: {vmsize_max} kB")
        print(f"Average VmSize: {vmsize_average} kB")
        print(f"Minimum VmRSS: {vmrss_min} kB")
        print(f"Maximum VmRSS: {vmrss_max} kB")
        print(f"Average VmRSS: {vmrss_average} kB")
        print(f"Minimum Data Read From Memory: {rchar_min} Bytes/s")
        print(f"Maximum Data Read From Memory: {rchar_max} Bytes/s")
        print(f"Average Data Read From Memory: {rchar_average} Bytes/s")
        print(f"Minimum Data Written To Memory: {wchar_min} Bytes/s")
        print(f"Maximum Data Written To Memory: {wchar_max} Bytes/s")
        print(f"Average Data Written To Memory: {wchar_average} Bytes/s")
        print("")
    for item in keys:
        gpu_timestamps = []
        gpu_utilization = []
        gpu_vram_utilization = []
        gpu_power_utilization = []
        for i in range(len(gpu_events[item])):
            gpu_timestamps.append(gpu_events[item][i].time_stamp)
            gpu_utilization.append(gpu_events[item][i].utilization)
            gpu_vram_utilization.append(gpu_events[item][i].ram_use)
            gpu_power_utilization.append(gpu_events[item][i].power_use)
        gpu_timestamp_min = min(gpu_timestamps)
        gpu_timestamp_max = max(gpu_timestamps)
        gpu_utilization_min = min(gpu_utilization)
        gpu_utilization_max = max(gpu_utilization)
        gpu_utilization_average = sum(gpu_utilization) / len(gpu_utilization)
        gpu_vram_min = min(gpu_vram_utilization)
        gpu_vram_max = max(gpu_vram_utilization)
        gpu_vram_average = sum(gpu_vram_utilization) / len(gpu_vram_utilization)
        gpu_power_min = min(gpu_power_utilization)
        gpu_power_max = max(gpu_power_utilization)
        gpu_power_average = sum(gpu_power_utilization) / len(gpu_power_utilization)
        gpu_driver_version = gpu_events[item][0].driver
        gpu_vram_system_max = gpu_events[item][0].ram_max
        gpu_power_system_max = gpu_events[item][0].power_max
        print("-------------------------------------------------------------------------------------------------------")
        print(f"Summary Of Usage Of GPU Resource: {item}")
        print("-------------------------------------------------------------------------------------------------------")
        print(f"GPU Driver Version: {gpu_driver_version}")
        print(f"Time On GPU: {gpu_timestamp_max - gpu_timestamp_min} Minutes")
        print(f"Minimum GPU Utilization: {gpu_utilization_min}")
        print(f"Maximum GPU Utilization: {gpu_utilization_max}")
        print(f"Average GPU Utilization: {gpu_utilization_average}")
        print(f"Minimum VRAM Utilization: {gpu_vram_min} / {gpu_vram_system_max} MB")
        print(f"Maximum VRAM Utilization: {gpu_vram_max} / {gpu_vram_system_max} MB")
        print(f"Average VRAM Utilization: {gpu_vram_average} / {gpu_vram_system_max} MB")
        print(f"Minimum Power Draw: {gpu_power_min} / {gpu_power_system_max} Watts")
        print(f"Maximum Power Draw: {gpu_power_max} / {gpu_power_system_max} Watts")
        print(f"Average Power Draw: {gpu_power_average} / {gpu_power_system_max} Watts")
        print("")


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
    cl_parser.add_argument('-i', '--interval', type=int, default=30,
                           help='Interval in minutes between labels on x axis. Default is 30.')
    cl_parser.add_argument('-n', '--no-legend', action='store_true', help='Remove the legend for generated plots.')
    cl_parser.add_argument('-r', '--rename', type=str, help='Base file name to rename generated plots to.')
    cl_parser.add_argument('-s', '--split', action='store_true', help='Split multiple GPUs out into separate plots.')
    cl_parser.add_argument('-p', '--sub-plot', action='store_true', help='Display GPU and CPU utilization as subplots.')
    cl_parser.add_argument('-o', '--offset', type=int, default=0, help='Time in hours to offset the GPU timing.')
    cl_parser.add_argument('--memprof-files', nargs='+', help='List of memprof.csv files. '
                                                              'Ex) memprof-42.csv memprof-1138.csv...')
    cl_parser.add_argument('-d', '--memprof-dir', type=str,
                           help='Directory containing the CSV files created by memprof.')
    cl_parser.add_argument('-c', '--plot-cpu', action='store_true',
                           help='Plot the CPU utilization on the GPU utilization plot.')
    cl_args = cl_parser.parse_args()
    logfile_name = cl_args.filename
    height = cl_args.height
    width = cl_args.width
    no_legend = cl_args.no_legend
    event_interval = cl_args.interval
    memprof_files = cl_args.memprof_files
    memprof_dir = cl_args.memprof_dir
    subplot = cl_args.sub_plot
    split_gpus = cl_args.split
    plot_cpu_utilization = cl_args.plot_cpu
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

    for event in events:
        event.time_stamp = event.time_stamp + datetime.timedelta(hours=cl_args.offset)
    if memprof_files is not None and memprof_dir is None:
        cpu_events = []
        for file in memprof_files:
            cpu_log_file = open(file, 'r')
            cpu_events = cpu_events + process_cpu_log(cpu_log_file)
            cpu_log_file.close()
        cpu_events = sorted(cpu_events, key=lambda x: x.time_stamp)
    elif memprof_files is None and memprof_dir is not None:
        cpu_events = []
        memprof_file = os.listdir(memprof_dir)
        for file in memprof_file:
            full_file_name = os.path.join(memprof_dir, file)
            if not full_file_name.endswith(".csv"):
                continue
            cpu_log_file = open(full_file_name, 'r')
            cpu_events = cpu_events + process_cpu_log(cpu_log_file)
            cpu_log_file.close()
        cpu_events = sorted(cpu_events, key=lambda x: x.time_stamp)
    elif memprof_files is not None and memprof_dir is not None:
        print("Please use either the cpu-files flag or memprof-dir flag. Not both.")
        sys.exit(1)
    else:
        cpu_events = None
    zero_time = get_zero_time(events, cpu_events)
    shift_gpu_event_times(events, zero_time)
    for event in events:
        if event.gpu_type in sorted_events:
            temp = sorted_events[event.gpu_type]
            temp.append(event)
            sorted_events[event.gpu_type] = temp
        else:
            temp = [event]
            sorted_events[event.gpu_type] = temp
    event_key_list = list(sorted_events.keys())
    if cpu_events is not None:
        shift_cpu_event_times(cpu_events, zero_time)
    create_plots_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name, cpu_events,
                      subplot, split_gpus, plot_cpu_utilization)

    print_statistics(sorted_events, event_key_list, cpu_events)


if __name__ == "__main__":
    main()
