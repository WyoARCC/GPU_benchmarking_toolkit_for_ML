#!/bin/bash
export PATH=/memory_profiler/memprof/bin:$PATH
if [ ! -d "./output" ]
then
   mkdir "./output"
fi
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Checking GPU!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
nvidia-smi
nvcc --version
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Checking Dataset!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

time memprof.sh bash scripts/download_dataset.sh &
PID=$!
echo "PID: $PID"
wait $PID

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Launching GPUSTAT!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
gpustat -a -i 1 > gpustat.log 2>&1 &
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Preparing Dataset!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh bash scripts/prepare_dataset.sh &
PID=$!	
echo "PID: $PID"	
wait $PID
RET_CODE=$?
if [ $RET_CODE != 0 ]
then
   exit $RET_CODE
fi
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Training!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export GRAD_ACCUMULATION=16
export EPOCS=250
time memprof.sh bash scripts/train.sh &
PID=$!	
echo "PID: $PID"	
wait $PID
RET_CODE=$?
if [ $RET_CODE != 0 ]
then
   exit $RET_CODE
fi
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Running Inference!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh bash scripts/inference_example.sh &
PID=$!	
echo "PID: $PID"	
wait $PID
RET_CODE=$?

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Script Finished!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
exit $RET_CODE
