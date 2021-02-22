#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

ssh -p 22334 root@testo-lang.ru "mkdir /var/www/testo-lang.ru/dist/v${1}"
scp -P 22334 $OUT_DIR/testo-cpu.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu.msi
scp -P 22334 $OUT_DIR/testo-gpu.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-gpu.msi
scp -P 22334 $OUT_DIR/testo-cpu.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu.deb
scp -P 22334 $OUT_DIR/testo-cpu.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu.rpm
scp -P 22334 $OUT_DIR/testo-gpu.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-gpu.deb
scp -P 22334 $OUT_DIR/testo-gpu.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-gpu.rpm
scp -P 22334 $OUT_DIR/testo-guest-additions-hyperv.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-hyperv.iso
scp -P 22334 $OUT_DIR/testo-guest-additions-qemu.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-qemu.iso
