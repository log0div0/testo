#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

ssh -p 22334 root@testo-lang.ru "mkdir -p /var/www/testo-lang.ru/dist/v${1}"

rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.deb
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.rpm
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.msi
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.deb
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.rpm
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.msi
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-guest-additions-qemu.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-qemu.iso
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-guest-additions-hyperv.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-hyperv.iso
