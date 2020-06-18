#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot \
	-o ${OUT_DIR}/testo-guest-additions.iso \
	${SCRIPT_DIR}/autorun.inf \
	${SCRIPT_DIR}/autorun.js \
	${TMP_DIR}/testo-guest-additions.deb \
	${TMP_DIR}/testo-guest-additions.rpm \
	${TMP_DIR}/testo-guest-additions-x64.msi \
	${TMP_DIR}/testo-guest-additions-x86.msi
