#!/bin/bash

export PATH=/memory_profiler/memprof/bin:$PATH
export OM_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
echo ""
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
if [ ! -d '../../LJSpeech-1.1' ]
then
WORKING_DIR=$(pwd)
cd '../..'
echo "The missing dataset will be retrived."
wget 'http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
tar -jxvf 'LJSpeech-1.1.tar.bz2'
cd "$WORKING_DIR"
fi 

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Launching GPUSTAT!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
gpustat -a -i 1 > gpustat.log 2>&1 &

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Training TTS Model Glow!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 TTS/bin/train_tts.py --config_path config.json &
PID=$!
echo "PID: $PID"
wait $PID
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Training VOCODER Model Higan!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 train_vocoder.py &
PID=$!
echo "PID: $PID"
wait $PID
RET_CODE=$?
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Training VOCODER Model Melgan!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

time memprof.sh python3 train_vocoder_melgan.py &
PID=$!
echo "PID: $PID"
wait $PID
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Script Finished!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
