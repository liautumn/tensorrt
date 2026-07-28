#!/usr/bin/env bash

set -euo pipefail

# 无论从哪个目录调用，都使用项目根目录作为 Docker 构建上下文。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"

dockerfile="${script_dir}/ubuntu-22.04.Dockerfile"
image_name="yolo26-tensorrt"
cuda_version="12.9.1"
tensorrt_version="10.13.3.9-1+cuda12.9"

show_help() {
  echo "用法：$0 [选项]"
  echo "  --file <路径>       Dockerfile 路径"
  echo "  --tag <镜像名>      生成的镜像名，默认 yolo26-tensorrt"
  echo "  --cuda <版本>       CUDA 基础镜像版本，默认 12.9.1"
  echo "  --tensorrt <版本>   TensorRT deb 版本，默认 10.13.3.9-1+cuda12.9"
  echo "  -h, --help          显示帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      dockerfile="$2"
      shift 2
      ;;
    --tag)
      image_name="$2"
      shift 2
      ;;
    --cuda)
      cuda_version="$2"
      shift 2
      ;;
    --tensorrt)
      tensorrt_version="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${dockerfile}" ]]; then
  echo "Dockerfile 不存在：${dockerfile}" >&2
  exit 1
fi

docker_args=(
  build
  --file "${dockerfile}"
  --build-arg "CUDA_VERSION=${cuda_version}"
  --build-arg "TENSORRT_VERSION=${tensorrt_version}"
  --build-arg "USER_ID=$(id -u)"
  --build-arg "GROUP_ID=$(id -g)"
  --tag "${image_name}"
  "${project_root}"
)

echo "正在构建镜像：${image_name}"
docker "${docker_args[@]}"
