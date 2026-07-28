#!/usr/bin/env bash

set -euo pipefail

# 固定挂载项目根目录，避免从 doc/docker 调用时挂载错误目录。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"

image_name="yolo26-tensorrt"
gpus="all"

show_help() {
  echo "用法：$0 [选项]"
  echo "  --tag <镜像名>      要启动的镜像，默认 yolo26-tensorrt"
  echo "  --gpus <值>         传给 Docker 的 GPU 范围，默认 all；none 表示不挂载 GPU"
  echo "  -h, --help          显示帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      image_name="$2"
      shift 2
      ;;
    --gpus)
      gpus="$2"
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

docker_args=(run --rm --interactive --tty)
if [[ "${gpus}" != "none" ]]; then
  docker_args+=(--gpus "${gpus}")
fi
docker_args+=(
  --volume "${project_root}:/workspace/yolo26"
  --workdir /workspace/yolo26
  "${image_name}"
)

docker "${docker_args[@]}"
