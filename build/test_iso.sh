#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

sudo testo run "$SCRIPT_DIR/../guest_additions/tests" \
	--stop_on_fail \
	--prefix tga_ \
	--param ISO_DIR "$ISO_DIR" \
	--param GUEST_ADDITIONS_ISO_PATH "$OUT_DIR/testo-guest-additions.iso" \
	--license "$LICENSE_PATH"
