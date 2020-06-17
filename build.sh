#!/usr/bin/env bash
set -euo pipefail

TESTO_VERSION="1.6.0"
SCRIPT_PATH=$(readlink -f "$0")
TESTO_SRC_DIR=$(dirname "$SCRIPT_PATH")
TESTO_DIST_DIR=$(readlink -f "$TESTO_SRC_DIR/out")
ISO_DIR=$(readlink -f "$TESTO_SRC_DIR/../iso")
LICENSE_PATH=$(readlink -f "$TESTO_SRC_DIR/../testo_license")

mkdir -p "$TESTO_DIST_DIR"

sudo testo run "$TESTO_SRC_DIR/build" \
	--stop_on_fail \
	--prefix tb_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_SRC_DIR "$TESTO_SRC_DIR" \
	--param TESTO_DIST_DIR "$TESTO_DIST_DIR" \
	--param TESTO_VERSION "$TESTO_VERSION" \
	--license "$LICENSE_PATH"