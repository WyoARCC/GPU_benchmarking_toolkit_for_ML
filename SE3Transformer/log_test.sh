#!/bin/bash

mn=$(hostname)
bash run_test.sh > "/GPU_logs/$mn-log.txt" 2>&1
