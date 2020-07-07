#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

sudo testo run "$SCRIPT_DIR/../tests/running_tests" \
	--stop_on_fail \
	--prefix tt_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_BUILD_DIR "$OUT_DIR" \
	--param TESTO_TESTS_DIR "$TESTO_TESTS_DIR" \
	--license "$LICENSE_PATH" \
	--assume_yes
