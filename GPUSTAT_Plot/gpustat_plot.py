import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import argparse
import datetime


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


def process_log(logfile):
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
                                           int(time_raw[0]), int(time_raw[1]), int(time_raw[2])).isoformat()
            utilization = usage_list[1].split()
            ram_usage = usage_list[2].split()
            events.append(
                gpu_usage_event(header_list[0], time_event, header_list[6], usage_list[0], utilization[0],
                                utilization[1], float(utilization[3]), float(utilization[6]), float(utilization[9]),
                                float(utilization[11]), float(utilization[13]), float(ram_usage[0]),
                                float(ram_usage[2])))
    return events


def create_plot(events, keys, height, width, event_interval, no_legend, save_name, plot_title, x_label, y_label,
                source):
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
    :param source:  What to generate the plot for. (GPU, POWER, or VRAM) Will default to VRAM
    """
    if source == "GPU":
        s = 1
    elif source == "POWER":
        s = 2
    else:
        s = 3
    fig, ax = plt.subplots(figsize=(width, height))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    if isinstance(keys, list):
        for item in keys:
            ax.plot(events[item][0], events[item][s], label=item)
    else:
        ax.plot(events[keys][0], events[keys][s], label=keys)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%D-%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=event_interval))
    ax.xaxis.get_ticklocs(minor=True)
    plt.minorticks_on()
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_name}_{source}_Utilization.png')


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
    cl_args = cl_parser.parse_args()
    logfile_name = cl_args.filename
    height = cl_args.height
    width = cl_args.width
    no_legend = cl_args.no_legend
    event_interval = cl_args.interval
    if cl_args.rename is not None:
        save_name = cl_args.rename
    else:
        save_name = cl_args.filename
    try:
        logfile_handle = open(logfile_name, 'r')
    except OSError:
        print(f"Could not open {logfile_name} please check the file path and permissions.")
        sys.exit(-1)
    events = process_log(logfile_handle)
    logfile_handle.close()
    # The GPU events are in chronological order but if multiple GPUs where used then that data will be mixed in.
    sorted_events = {}
    for event in events:
        gpu_id = event.gpu_type
        if gpu_id not in sorted_events:
            sorted_events.update({gpu_id: [[], [], [], []]})
        sub_list = sorted_events[gpu_id]
        times = sub_list[0]
        times.append(np.datetime64(event.time_stamp))
        gpu_utilization = sub_list[1]
        gpu_utilization.append(event.utilization)
        power_utilization = sub_list[2]
        power_utilization.append(event.power_use)
        vram_utilization = sub_list[3]
        vram_utilization.append(event.ram_use)
        sorted_events[gpu_id] = sub_list
    event_key_list = sorted(sorted_events)

    if cl_args.split is True:
        for i in range(len(event_key_list)):
            file_name = f"{save_name}_{i}"
            # Plot percent GPU utilization.
            plot_name = f"{i}_GPU Utilization"
            xlabel = "Time"
            ylabel = "Percent GPU Utilization"
            source = "GPU"
            create_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                        plot_name, xlabel, ylabel, source)
            # Plot power utilization.
            plot_name = f"{i}_Power Utilization"
            ylabel = f"{i}_GPU Power Utilization"
            source = "POWER"
            create_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                        plot_name, xlabel, ylabel, source)
            # Plot VRAM utilization.
            plot_name = f"{i}_VRAM Utilization"
            ylabel = f"{i}_VRAM Utilization"
            source = "VRAM"
            create_plot(sorted_events, event_key_list[i], height, width, event_interval, no_legend, file_name,
                        plot_name, xlabel, ylabel, source)
    else:
        # Plot percent GPU utilization.
        plot_name = "GPU Utilization"
        xlabel = "Time"
        ylabel = "Percent GPU Utilization"
        source = "GPU"
        create_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                    plot_name, xlabel, ylabel, source)
        # Plot power utilization.
        plot_name = "GPU Power Utilization"
        ylabel = "GPU Power Utilization"
        source = "POWER"
        create_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                    plot_name, xlabel, ylabel, source)
        # Plot VRAM utilization.
        plot_name = "VRAM Utilization"
        ylabel = "VRAM Utilization"
        source = "VRAM"
        create_plot(sorted_events, event_key_list, height, width, event_interval, no_legend, save_name,
                    plot_name, xlabel, ylabel, source)


if __name__ == "__main__":
    main()
