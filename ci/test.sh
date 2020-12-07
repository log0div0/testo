#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

if [[ $# -eq 0 ]] ; then
	TEST_SPEC=""
else
	TEST_SPEC="--test_spec $1"
fi

sudo testo run "$SCRIPT_DIR/test_scripts" \
	--stop_on_fail \
	--prefix tt_ \
	--param ISO_DIR "$ISO_DIR" \
	--param OUT_DIR "$OUT_DIR" \
	--param TEST_ASSETS_DIR "$SCRIPT_DIR/test_assets" \
	--license "$LICENSE_PATH" \
	${TEST_SPEC} \
	--assume_yes
