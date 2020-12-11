#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

if [ ! -e "${ONNXRUNTIME_SRC_DIR}" ]; then
	git clone https://github.com/microsoft/onnxruntime.git "${ONNXRUNTIME_SRC_DIR}"
fi

cd "${ONNXRUNTIME_SRC_DIR}"
git checkout v1.3.0
git submodule sync --recursive
git submodule update --init --recursive
