#!/bin/bash

#SBATCH --account=arcc-students
#SBATCH --time=12:00:00
#SBATCH --job-name=GPUBenchMarkGPT2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tlimato@uwyo.edu
#SBATCH --output=GPUBenchMarkGPT2_%A.log
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:2

$SLURM_JOB_ID
echo "Load Modules:"
module load miniconda3/4.12.0
module load sarcc/1.0
module load gcc/11.2.0
module load cuda/11.8.0
echo "Modules Loaded"
echo "Activate Conda Environment:"
conda activate /pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/GPT2Script
echo "Creating Environment Variables:"
#Equal to Number of GPUs Requested
export WORLD_SIZE=2
#Automatically Handled by SLURM
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=0,1  # Set the IDs of the GPUs to use
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"  # Print out the GPUs being used
echo "Environmental Variables Initialized."

python torch.distributed.launch main.py

conda deactivate
