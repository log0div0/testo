#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

for HYPERVISOR in qemu hyperv; do
	sudo genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot \
		-o ${OUT_DIR}/testo-guest-additions-${HYPERVISOR}.iso \
		${SCRIPT_DIR}/build_assets/autorun.inf \
		${SCRIPT_DIR}/build_assets/autorun.js \
		${TMP_DIR}/${HYPERVISOR}/testo-guest-additions.deb \
		${TMP_DIR}/${HYPERVISOR}/testo-guest-additions.rpm \
		${TMP_DIR}/${HYPERVISOR}/testo-guest-additions-x64.msi \
		${TMP_DIR}/${HYPERVISOR}/testo-guest-additions-x86.msi
done
