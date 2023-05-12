ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.10-py3
FROM ${FROM_IMAGE_NAME}

# ARG UNAME
# ARG UID
# ARG GID
# RUN groupadd -g $GID -o $UNAME
# RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# USER $UNAME

ADD . /workspace/fastspeech
ADD log_test.sh /log_test.sh
WORKDIR /workspace/fastspeech
RUN sh ./scripts/install.sh
RUN wget 'https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/20.01.0/files/nvidia_waveglow256pyt_fp16.pt'
RUN wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA&export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA" -O tacotron2_statedict.pt && rm -rf /tmp/cookies.txt
WORKDIR /
RUN mkdir memory_profiler
WORKDIR /memory_profiler 
RUN git clone https://github.com/WyoARCC/memprof
WORKDIR /
RUN pip uninstall pynvml -y
RUN pip --no-cache-dir install pytools==2022.1.12 gpustat
