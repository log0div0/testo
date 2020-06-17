#!/usr/bin/env bash
set -euo pipefail

TESTO_VERSION="1.6.0"
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
TESTO_SRC_DIR=$(readlink -f "$SCRIPT_DIR/..")
OUT_DIR=$(readlink -f "$SCRIPT_DIR/out")
TMP_DIR=$(readlink -f "$SCRIPT_DIR/tmp")
ISO_DIR=$(readlink -f "$TESTO_SRC_DIR/../iso")
LICENSE_PATH=$(readlink -f "$TESTO_SRC_DIR/../testo_license")

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_DIR"

sudo testo run "$SCRIPT_DIR/src" \
	--stop_on_fail \
	--prefix tb_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_SRC_DIR "$TESTO_SRC_DIR" \
	--param TMP_DIR "$TMP_DIR" \
	--param OUT_DIR "$OUT_DIR" \
	--param TESTO_VERSION "$TESTO_VERSION" \
	--license "$LICENSE_PATH" \
	--test_spec $1
