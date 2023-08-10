#!/bin/bash 

GPU_MODEL=$(lspci | grep -m1 NVIDIA | grep -oP '\[\K[^]]+' | cut -d ' ' -f 1)

# gpustat
gpustat -a -i 2 --no-header > "memprof/gpustat_${GPU_MODEL}_$(date +%Y%m%d%H%M%S).log" 2>&1 &

# sar command
sar -r 2 > "memprof/memstat_${GPU_MODEL}_$(date +%Y%m%d%H%M%S).log" 2>&1 &
sar -u 2 > "memprof/cpustat_${GPU_MODEL}_$(date +%Y%m%d%H%M%S).log" 2>&1 &

# memprof based on checking script (not always finding the correct PID)
time memprof.sh bash train_script.sh

# memprof based on finding PID ourselves (not always working)
#bash train_script.sh &
# getting PID of last run command
#PID=$!
#time memprof_PID.sh $PID

kill $(pgrep sar)
kill $(pgrep gpustat)

