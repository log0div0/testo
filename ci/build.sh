#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_DIR"

if [[ $# -eq 0 ]] ; then
	TEST_SPEC=""
else
	TEST_SPEC="--test_spec $1"
fi

sudo testo run "$SCRIPT_DIR/build_scripts" \
	--stop_on_fail \
	--prefix tb_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_SRC_DIR "$TESTO_SRC_DIR" \
	--param BUILD_ASSETS_DIR "$SCRIPT_DIR/build_assets" \
	--param TMP_DIR "$TMP_DIR" \
	--param OUT_DIR "$OUT_DIR" \
	--license "$LICENSE_PATH" \
	${TEST_SPEC} \
	--assume_yes
