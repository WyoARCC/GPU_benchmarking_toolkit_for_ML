#!/bin/bash
# Helper script to get the dataset, preprocess it, train, and generate some audio files using the trained TTS.

export PATH=/memory_profiler/memprof/bin:$PATH
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
echo "Launching GPUSTAT!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
gpustat -a -i 60 > gpustat.log 2>&1 &
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Getting Data!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh bash scripts/prepare_dataset.sh &
PID=$!
echo "PID: $PID"
wait $PID

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Preprocess Data!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 fastspeech/dataset/ljspeech_dataset.py --dataset_path="./LJSpeech-1.1" --mels_path="./mels_ljspeech1.1" &
PID=$!
echo "PID: $PID"
wait $PID

echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Preprocess Alignment!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 fastspeech/align_tacotron2.py --dataset_path="./LJSpeech-1.1" --tacotron2_path="tacotron2_statedict.pt" --aligns_path="aligns_ljspeech1.1" &
PID=$!
echo "PID: $PID"
wait $PID

export CUDA_VISIBLE_DEVICES=0


echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Running Training!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 fastspeech/train.py --dataset_path="./LJSpeech-1.1" --mels_path="./mels_ljspeech1.1" --aligns_path="./aligns_ljspeech1.1" --log_path="./logs" --checkpoint_path="./checkpoints" &
PID=$!
echo "PID: $PID"
wait $PID


echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Generating!"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
time memprof.sh python3 generate.py --waveglow_path=nvidia_waveglow256pyt_fp16.pt --text=test_sentences.txt &
PID=$!
echo "PID: $PID"
wait $PID

