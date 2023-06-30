RUN echo "Load Modules:"
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
# ONLY RUN ONCE WHEN CONFIGURING CONTAINER
echo "create conda environment"

conda create -n GPT2_NRP python=3.10

echo "Activate Conda Environment:"
# Make RUN commands use the new environment:
conda activate /pfs/tc1/project/arcc-students/tlimato/GPT2_SHDev/GPT2Script
# ONE TIME INSTALL PACKAGES
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install accelerate
pip install numpy
pip install matplotlib