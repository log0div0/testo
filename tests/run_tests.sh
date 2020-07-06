#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

sudo testo run "$SCRIPT_DIR/src" \
	--stop_on_fail \
	--prefix tt_ \
	--param ISO_DIR "$ISO_DIR" \
	--param TESTO_BUILD_DIR "$TESTO_BUILD_DIR" \
	--license "$LICENSE_PATH" 
