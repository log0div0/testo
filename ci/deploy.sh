#!/usr/bin/env bash

source "$(dirname "$0")/vars.sh"

ssh -p 22334 root@testo-lang.ru "mkdir -p /var/www/testo-lang.ru/dist/v${1}"
ssh -p 22334 root@testo-lang.ru "mkdir -p /var/www/testo-lang.ru/dist/v${1}/arm"

rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.deb
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.rpm
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-${1}.msi
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.deb
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.rpm root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.rpm
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-nn-server.msi root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-nn-server-${1}.msi
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-guest-additions-qemu.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-qemu.iso
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/testo-guest-additions-hyperv.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/testo-guest-additions-${1}-hyperv.iso

rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/arm/testo.deb root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/arm/testo-${1}.deb
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/arm/testo-guest-additions-qemu.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/arm/testo-guest-additions-${1}-qemu.iso
rsync -e "ssh -p 22334" -avz --info=progress2 $OUT_DIR/arm/testo-guest-additions-hyperv.iso root@testo-lang.ru:/var/www/testo-lang.ru/dist/v${1}/arm/testo-guest-additions-${1}-hyperv.iso

echo "
Windows client (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-${1}.msi
Windows server (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-nn-server-${1}.msi
Hyper-V guest additions (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-guest-additions-${1}-hyperv.iso
Hyper-V guest additions (arm64): https://testo-lang.ru/storage/dist/v${1}/arm/testo-guest-additions-${1}-hyperv.iso

RPM client (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-${1}.rpm
RPM server (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-nn-server-${1}.rpm

DEB client (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-${1}.deb
DEB client (arm64): https://testo-lang.ru/storage/dist/v${1}/arm/testo-${1}.deb
DEB server (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-nn-server-${1}.deb
QEMU guest additions (x86_64): https://testo-lang.ru/storage/dist/v${1}/testo-guest-additions-${1}-qemu.iso
QEMU guest additions (arm64): https://testo-lang.ru/storage/dist/v${1}/arm/testo-guest-additions-${1}-qemu.iso
" > $OUT_DIR/links.txt

cat $OUT_DIR/links.txt
