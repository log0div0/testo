#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

ssh root@testo-lang.ru "mkdir /var/www/testo-lang.ru/dist/v${1}"
scp $OUT_DIR/testo-cpu-x64.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu-x64.msi
scp $OUT_DIR/testo-cpu-x86.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu-x86.msi
scp $OUT_DIR/testo-cpu.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu.deb
scp $OUT_DIR/testo-cpu.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-cpu.rpm
scp $OUT_DIR/testo-gpu.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-gpu.deb
scp $OUT_DIR/testo-gpu.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}-gpu.rpm
scp $OUT_DIR/testo-guest-additions.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}.iso
