#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=12.8.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
LABEL maintainer="NVIDIA CORPORATION"

ENV NV_CUDNN_VERSION 8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV CUDA_VERSION_MAJOR_MINOR=12.2

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"

ENV TRT_VERSION 10.9.0.34
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install requried libraries
# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    valgrind \
    gdb \
    cmake \
    libopencv-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make

# Install TensorRT
COPY TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz /tmp/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz
# 解压并安装到 /usr/lib/x86_64-linux-gnu
RUN tar -xf /tmp/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz -C /tmp/ \
    && cp -a /tmp/TensorRT-10.9.0.34/lib/*.so* /usr/lib/x86_64-linux-gnu/ \
    && rm -rf /tmp/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu

ENV CUDA_PATH /usr/local/cuda-12.8
ENV PATH $CUDA_PATH/bin:$PATH
ENV LD_LIBRARY_PATH $CUDA_PATH/lib64:$LD_LIBRARY_PATH

ENV TENSORRT_PATH=/home/autumn/Documents/CUDA/TensorRT-10.9.0.34
ENV PATH=$TENSORRT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
