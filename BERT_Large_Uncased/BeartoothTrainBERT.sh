#!/bin/bash

#SBATCH --account=arcc-students
#SBATCH --time=60:00:00
#SBATCH --job-name=GPUBenchMarkBERTA30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tlimato@uwyo.edu
#SBATCH --output=GPUBenchMarkBERT_%A.log
#SBATCH --gres=gpu:2
#SBATCH --partition=beartooth-gpu


echo $SLURM_JOB_ID
echo "Load Modules:"
module load miniconda3/4.12.0
module load gcc/11.2.0
module load cuda/11.8.0
echo "Modules Loaded"
echo "Cuda Devices"
srun nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"  # Print out the GPUs being used

echo "Setting HuggingFace Cache Directories..."
export HF_DATASETS_CACHE="/pfs/tc1/project/arcc-students/tlimato/BERT_SHDev/Datasets_cache/"
export TRANSFORMERS_CACHE="/pfs/tc1/project/arcc-students/tlimato/BERT_SHDev/Transformer_cache/"

echo "Activate Conda Environment:"
conda activate /pfs/tc1/project/arcc-students/tlimato/BERT_SHDev/BERT_Testing

echo "Creating Environment Variables:"
export BATCH_TRAIN_SIZE=32
export MODEL_NAME="bert-large-uncased"
export MODEL_TOKENIZER="bert-large-uncased"
export PRECISION_TRAIN="fp16"
export BM_INTERVAL=1
export NUM_EPOCHS=1
echo "Environmental Variables Initialized."

gpustat -a -i 30 > gpustat.log 2>&1 &
echo "Starting LLM Benchmark"
accelerate launch main.py

python Plot_nonGPU_functions.py
conda deactivate

python utilization_plot.py gpustat.log -i 25
