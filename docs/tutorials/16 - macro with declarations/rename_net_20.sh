#!/bin/bash

set -e

mac=$1

oldname=`ip -o link | grep ${mac,,} | awk '{print substr($2, 1, length($2) - 1)}'`
newname=$2

echo SUBSYSTEM==\"net\", ACTION==\"add\", ATTR{address}==\"$mac\", NAME=\"$newname\", DRIVERS==\"?*\" >> /lib/udev/rules.d/70-test-tools.rules

rm -f /etc/network/interfaces
echo source /etc/network/interfaces.d/* >> /etc/network/interfaces
echo auto lo >> /etc/network/interfaces
echo iface lo inet loopback >> /etc/network/interfaces

ip link set $oldname down
ip link set $oldname name $newname
ip link set $newname up

echo "Renaming success"
