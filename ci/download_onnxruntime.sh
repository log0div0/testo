#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

if [ ! -e "${ONNXRUNTIME_SRC_DIR}" ]; then
	git clone https://github.com/microsoft/onnxruntime.git "${ONNXRUNTIME_SRC_DIR}"
fi

cd "${ONNXRUNTIME_SRC_DIR}"
git checkout .
git checkout v1.5.3
git submodule sync --recursive
git submodule update --init --recursive
git apply $SCRIPT_DIR/build_assets/onnxruntime_patches/single_thread.patch
git apply $SCRIPT_DIR/build_assets/onnxruntime_patches/cuda.patch
