FROM nvidia/cuda:11.7.1-cudnn8-runtime-rockylinux8
RUN dnf module install python38 -y
RUN dnf update -y && \
    dnf install -y  cuda-toolkit-11-7 make git cmake python38-devel libsndfile lapack pigz

WORKDIR /
RUN mkdir memory_profiler
WORKDIR /memory_profiler
RUN git clone https://github.com/IGBIllinois/memprof
WORKDIR /
COPY . /workspace/tts
WORKDIR /workspace/tts
RUN pip3 --no-cache-dir install --upgrade pip
RUN pip3 --no-cache-dir install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 --no-cache-dir install -e .[all,dev,notebooks]
RUN pip3 --no-cache-dir install TTS
RUN make install
RUN pip3 uninstall pynvml -y
RUN pip3 --no-cache-dir install gpustat
