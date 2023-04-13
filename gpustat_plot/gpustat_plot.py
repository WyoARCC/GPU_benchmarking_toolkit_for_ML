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
    # Setting an interval to use for the number of GPU events to limit the number that is displayed.
    # Plot GPU utilization.
    fig, ax = plt.subplots(figsize=(width, height))

    plt.ylabel("Percent GPU Utilization")
    plt.xlabel("Time")
    plt.title("GPU Utilization Over Time")
    for item in event_key_list:
        ax.plot(sorted_events[item][0], sorted_events[item][1], label=item)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%D-%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=event_interval))
    ax.xaxis.get_ticklocs(minor=True)
    plt.minorticks_on()
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_name}_GPU_Utilization.png')

    # Plot Power utilization.
    fig, ax = plt.subplots(figsize=(width, height))
    plt.ylabel("Power Consumed By GPU (Watts)")
    plt.xlabel("Time")
    plt.title("Power Consumption Over Time")
    for item in event_key_list:
        ax.plot(sorted_events[item][0], sorted_events[item][2], label=item)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%D-%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=event_interval))
    ax.xaxis.get_ticklocs(minor=True)
    plt.minorticks_on()
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_name}_Power_Utilization.png')

    # Plot VRAM utilization.
    fig, ax = plt.subplots(figsize=(width, height))
    plt.ylabel("VRAM Utilized (Megabytes)")
    plt.xlabel("Time")
    plt.title("VRAM Utilization Over Time")
    for item in event_key_list:
        ax.plot(sorted_events[item][0], sorted_events[item][3], label=item)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%D-%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=event_interval))
    ax.xaxis.get_ticklocs(minor=True)
    plt.minorticks_on()
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    if no_legend is False:
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_name}_VRAM_Utilization.png')


if __name__ == "__main__":
    main()
