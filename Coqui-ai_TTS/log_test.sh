#!/bin/bash

mn=$(hostname)
bash run_test.sh > "/GPU_logs/$mn-log.txt" 2>&1
mv ./memprof "/GPU_logs/$mn-memprof"
mv gpustat.log "/GPU_logs/$mn-gpustat.log"
