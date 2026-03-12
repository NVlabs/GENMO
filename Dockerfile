# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    gosu \
    xvfb \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set up workspace
WORKDIR /workspace/gem-smpl
COPY . /workspace/gem-smpl

# Create virtual environment
RUN uv venv .venv --python 3.10

# Install PyTorch
RUN . .venv/bin/activate && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install GEM-SMPL and dependencies
RUN . .venv/bin/activate && \
    bash scripts/install_env.sh

# Headless rendering environment
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless

# Activate venv by default
ENV PATH="/workspace/gem-smpl/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/workspace/gem-smpl/.venv"

ENTRYPOINT ["tools/docker-entrypoint.sh"]
CMD ["bash"]
