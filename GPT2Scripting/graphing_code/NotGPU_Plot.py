from Plot_nonGPU_functions import plot_training_loss, plot_epoch_time, plot_throughput, plot_disk_read_iops, \
    plot_disk_write_iops, plot_disk_iops, plot_cpu_utilization, plot_thread_count, plot_ram_utilization_percent

# Plot Training Loss and save the image
plot_training_loss('training_results.csv', 'training_loss.png')
# Plot Epoch Time and save the image
plot_epoch_time('training_results.csv', 'epoch_time.png')
# Plot Throughput and save the image
plot_throughput('training_results.csv', 'throughput.png')
# Plot Disk Read IOPS and save the image
plot_disk_read_iops('training_results.csv', 'disk_read_iops.png')
# Plot Disk Write IOPS and save the image
plot_disk_write_iops('training_results.csv', 'disk_write_iops.png')
# Plot Disk IOPS and save the image
plot_disk_iops('training_results.csv', 'disk_iops.png')
# Plot CPU Utilization and save the image
plot_cpu_utilization('CPU_RAM_Utilization.csv', 'cpu_utilization_percent.png')
# Plot Thread Count and save the image
plot_thread_count('CPU_RAM_Utilization.csv', 'thread_count.png')
# Plot RAM Utilization and save the image
plot_ram_utilization_percent('CPU_RAM_Utilization.csv', 'ram_utilization_percent.png')
