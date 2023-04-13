# gpustat_plot
This python script reads a dump of the output of [gpustat](https://github.com/wookayin/gpustat) and generates plots for GPU utilization, VRAM utilization, and GPU power consumption over time.

## Table Of Contents
- [Release notes](#release-notes)
    * [Usage](#usage)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Release notes
Requires [matplotlib](https://pypi.org/project/matplotlib) to run. 
## Usage
To run the application clone the Repo then run the script on the file.
```
git clone https://github.com/WyoARCC/GPU_benchmarking_toolkit_for_ML
cd GPU_benchmarking_toolkit_for_ML/gpustat_plot/
python3 gpustat_plot.py /location/of/file.log
```
This will generate the three utilization graphs in the same directory as the file specified to the script. These utilization graphs will be named after the file used to create them. As an example, a file named file1.log will produce three graphs: file1.log_GPU_Utilization.png, file1.log_Power_Utilization.png, and file1.log_VRAM_Utilization.png.
The example files are generated with the following command:
```
python3 gpustat_plot.py ./example/se3-transformer-job-rtx2080-jvdgz-gpustat.log -i 1020
```
### Changelog
2023/04/13
- Added command line flags to allow customization of the generated plots from the command line.

2023/04/11
- Updated code to display legend and updated the example files to display the legend.
- Updated code to dynamically set interval for the x axis in generated plots.

2023/04/10
- Initial commit.

### Known issues

There are no known issues in this release.
