#!/bin/bash

# Create conda environment in current directory
load module load miniconda3/4.12.0
conda create -n GPT2Scripting python=3.10
# Activate conda environment
conda activate GPT2Scripting

# Install packages from requirements.txt
pip install -r requirements.txt

# Deactivate Conda Environment
conda deactivate
