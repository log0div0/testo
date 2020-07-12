#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

sudo testo run "$SCRIPT_DIR/test_scripts" \
	--stop_on_fail \
	--prefix tt_ \
	--param ISO_DIR "$ISO_DIR" \
	--param OUT_DIR "$OUT_DIR" \
	--param TEST_ASSETS_DIR "$SCRIPT_DIR/test_assets" \
	--license "$LICENSE_PATH" \
	--test_spec $1 \
	--assume_yes
