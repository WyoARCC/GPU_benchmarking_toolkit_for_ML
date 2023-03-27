#!/bin/bash
# Helper script for running the tests and preserving the output in a log file. The default size of the workspace available on the Kubernetes cluster that was being worked with was insufficient for running the model so a larger temporary space was mounted when launching the Kubernetes job and the existing files are moved to this larger space and the benchmark is run from there.

mv /workspace/fastspeech /larger_workspace
cd /larger_workspace/fastspeech
mn=$(hostname)
bash run_test.sh > "/GPU_logs/$mn-log.txt" 2>&1
