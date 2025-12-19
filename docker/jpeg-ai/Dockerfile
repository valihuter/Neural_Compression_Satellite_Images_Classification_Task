# JPEG-AI Reference Software Docker Image
# Based on Ubuntu 22.04 with CUDA support

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Clone JPEG-AI repository (skip LFS during clone, fetch separately)
WORKDIR /workspace
RUN git lfs install
# Clone without LFS files first, then fetch them
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git
WORKDIR /workspace/jpeg-ai-reference-software

# Fetch LFS files with retry logic
RUN git lfs pull || (sleep 5 && git lfs pull) || (sleep 10 && git lfs pull) || echo "Warning: LFS pull failed, models may need manual download"

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    pybind11 \
    einops \
    opencv-python \
    pillow \
    numpy \
    scipy \
    scikit-image \
    pandas \
    tqdm \
    matplotlib \
    pyyaml \
    commentjson \
    nvidia-ml-py \
    attrs \
    pytorch-msssim \
    addict \
    prettytable \
    ptflops

# Build C++ extensions
WORKDIR /workspace/jpeg-ai-reference-software/src/codec/entropy_coding/cpp_exts/mans
RUN make
WORKDIR /workspace/jpeg-ai-reference-software/src/codec/entropy_coding/cpp_exts/direct
RUN make

# Copy .so files to lib_wrappers
WORKDIR /workspace/jpeg-ai-reference-software
RUN cp src/codec/entropy_coding/cpp_exts/mans/*.so src/codec/entropy_coding/lib_wrappers/mans/ || true && \
    cp src/codec/entropy_coding/cpp_exts/direct/*.so src/codec/entropy_coding/lib_wrappers/direct/ || true

# Set working directory
WORKDIR /workspace/jpeg-ai-reference-software

# Default command
CMD ["/bin/bash"]
