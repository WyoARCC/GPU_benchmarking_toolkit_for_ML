#!/bin/bash

#SBATCH --account=arcc-students -
#SBATCH --time=12:00:00
#SBATCH --job-name=GPUBenchMarkGPT2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tlimato@uwyo.edu
#SBATCH --output=GPUBenchMarkGPT2_%A.log
#SBATCH --gres=gpu:2
#SBATCH --partition=beartooth-gpu

echo "Load Modules:"
module load miniconda3/4.12.0
echo "Activate Conda Environment"
conda activate /pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/GPT2scripting
python TrainGPT2WikiText.py

conda deactivate

seff jobid
