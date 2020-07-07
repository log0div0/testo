#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_DIR"

sudo testo run "$SCRIPT_DIR/src" \
	--stop_on_fail \
	--prefix tb_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_SRC_DIR "$TESTO_SRC_DIR" \
	--param TMP_DIR "$TMP_DIR" \
	--param OUT_DIR "$OUT_DIR" \
	--license "$LICENSE_PATH" \
	--test_spec $1 \
	--assume_yes
