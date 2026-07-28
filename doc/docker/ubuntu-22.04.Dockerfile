# CUDA 和 TensorRT 版本可由 build.sh 参数覆盖，两者必须使用兼容的构建版本。
ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TENSORRT_VERSION=10.13.3.9-1+cuda12.9

LABEL org.opencontainers.image.title="YOLO26 TensorRT build environment"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# TensorRT 直接从 NVIDIA CUDA 软件源安装，并固定为 10.x，避免仓库升级到 11.x
# 后与项目的 TensorRT 10 接口不兼容。
# OpenCV 只用于构建 workspace/example，yolo26 动态库不链接 OpenCV。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       cmake \
       git \
       "libnvinfer-dev=${TENSORRT_VERSION}" \
       "libnvinfer10=${TENSORRT_VERSION}" \
       libopencv-dev \
       ninja-build \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 使用宿主机 UID/GID 写入挂载目录，生成的 build 文件无需额外修改权限。
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN if getent group "${GROUP_ID}" >/dev/null; then \
         build_group="$(getent group "${GROUP_ID}" | cut -d: -f1)"; \
       else \
         groupadd --gid "${GROUP_ID}" yolo26; \
         build_group="yolo26"; \
       fi \
    && useradd --non-unique --uid "${USER_ID}" --gid "${build_group}" \
       --create-home --shell /bin/bash yolo26 \
    && mkdir -p /workspace/yolo26 \
    && chown -R yolo26:"${build_group}" /workspace/yolo26

ENV CUDA_PATH=/usr/local/cuda
ENV TENSORRT_ROOT=/usr
WORKDIR /workspace/yolo26
USER yolo26

CMD ["/bin/bash"]
