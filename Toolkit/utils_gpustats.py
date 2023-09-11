import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_gpu_log(logfile, logging_frequency_s, gpu_nums):
    gpu_data = []

    for i, line in enumerate(logfile):
        if line.startswith("["):
            data = line.split("|")
            gpu_id = int(data[0].strip()[1])

            temperature = int(data[1].split(",")[0].strip().split()[0].replace('°C', ''))
            utilization_info = data[1].split(",") # 24°C,  ?? %,   0 % (E:   0 %  D:   0 %),   27 / 165 W
            gpu_utilization = int(utilization_info[2].strip().split("%")[0])
            power_usage = int(utilization_info[3].strip().split("/")[0])

            memory_usage = int(data[2].strip().split("/")[0])

            timestamp = np.floor(i/gpu_nums) * logging_frequency_s

            gpu_data.append({
                'GPU ID': gpu_id,
                'Timestamp': timestamp,
                'Temperature (°C)': temperature,
                'GPU Utilization (%)': gpu_utilization,
                'Power Usage (W)': power_usage,
                'Memory Usage (MB)': memory_usage
            })

    gpu_dfs = []
    for gpu_id in range(gpu_nums):
        gpu_df = pd.DataFrame([data for data in gpu_data if data['GPU ID'] == gpu_id])
        gpu_dfs.append(gpu_df)

    return gpu_dfs


def plot_gpu_utilization(gpu_dfs, model_name, log_folder, log_path):
    gpu_nums = len(gpu_dfs)

    # Create a figure for GPU Utilization
    plt.figure(figsize=(5*gpu_nums, 5))
    plt.suptitle(f'GPU (%) Utilization ({gpu_nums} GPUs) | Model: {model_name} | {log_folder}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i+1)
        plt.scatter(gpu_df['Timestamp']/60, gpu_df['GPU Utilization (%)'], label=f'GPU {i+1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('GPU Utilization (%)')
        plt.ylim(-5, 100)
        plt.axhline(y=gpu_df['GPU Utilization (%)'].mean(), color='r', linestyle='--', label='Average Utilization')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(f'{log_path}/{model_name}_{log_folder}_gpu_utilization.png')

    # Create a separate figure for Power Usage
    plt.figure(figsize=(5*gpu_nums, 5))
    plt.suptitle(f'GPU Power Utilization (spec power 165W) ({gpu_nums} GPUs) | Model: {model_name} | {log_folder}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i+1)
        plt.scatter(gpu_df['Timestamp']/60, gpu_df['Power Usage (W)'], label=f'GPU {i+1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power Usage (W)')
        plt.ylim(-5, 200)
        plt.axhline(y=gpu_df['Power Usage (W)'].mean(), color='r', linestyle='--', label='Average Power Usage')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(f'{log_path}/{model_name}_{log_folder}_gpu_power_usage.png')

    # Create a separate figure for Memory Utilization
    plt.figure(figsize=(5*gpu_nums, 5))
    plt.suptitle(f'GPU Memory Utilization (total memory 24576 MB) ({gpu_nums} GPUs) | Model: {model_name} | {log_folder}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i+1)
        plt.scatter(gpu_df['Timestamp']/60, gpu_df['Memory Usage (MB)'], label=f'GPU {i+1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory Usage (MB)')
        plt.ylim(-100, 24576)
        plt.axhline(y=gpu_df['Memory Usage (MB)'].mean(), color='r', linestyle='--', label='Average Memory Usage (MB)')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(f'{log_path}/{model_name}_{log_folder}_gpu_memory_usage.png')

    # Display the entire plot
    plt.show()
