#!/bin/bash

#SBATCH --account=arcc-students
#SBATCH --time=95:00:00
#SBATCH --job-name=GPUBenchMarkGPT2_V100_32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tlimato@uwyo.edu
#SBATCH --output=GPUBenchMarkGPT2_%A.log
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --nodelist=tdgx01


echo $SLURM_JOB_ID

echo "Loading modules Modules:"
module load miniconda3/4.12.0
module load gcc/11.2.0
module load cuda/11.8.0
echo "Modules Loaded"

echo "Cuda Devices"
srun nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"  # Print out the GPUs being used

echo "Setting HuggingFace Cache Directories..."
export HF_DATASETS_CACHE="/pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/Datasets_cache/"
export TRANSFORMERS_CACHE="/pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/Transfomer_cache/"

echo "Activate Conda Environment:"
conda activate /pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/GPT2Script

echo "Creating Environment Variables:"
export BATCH_TRAIN_SIZE=16
export MODEL_NAME="gpt2-large"
export MODEL_TOKENIZER="gpt2-large"
export PRECISION_TRAIN="fp16"
export TASK_TYPE="finetune"
export BM_INTERVAL="1"
export NUM_EPOCHS="1"
echo "Environmental Variables Initialized."

gpustat -a -i 30 > gpustat.log 2>&1 &
echo "Starting LLM Benchmark"
accelerate launch main.py

echo "Generating Graphs"
python Plot_nonGPU_functions.py

#call plotone_pretty.sh memprof-810497.csv to generate plot with appropriate file name
conda deactivate
