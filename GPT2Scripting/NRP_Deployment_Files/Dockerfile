# Author: Tyson Limato
# project: GPU Benchmarking
# Model: GPT2 Variations from HuggingFace
# Backend: Pytorch
# Use the base miniconda3 image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the Docker image
COPY environment.yml .

# Create a new environment and install dependencies
RUN conda env create --file environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "gpt2_nrp", "/bin/bash", "-c"]

# Copy your application code to the Docker image
COPY main.py /app
COPY Plot_nonGPU_functions.py /app
COPY DataSetsForLLM.py /app
COPY start.sh /app
# Make start.sh executable
RUN chmod +x /app/start.sh

# Run start.sh script when the container starts
CMD ["/bin/bash", "/app/start.sh"]
