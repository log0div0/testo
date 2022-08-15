# Getting started

This page contains the instructions to prepare your system for Testo Framework usage. If you have any troubles or difficulties with setting Testo up, or if you just have any questions - please contact us on support@testo-lang.ru

# Installing testo-nn-server

You should start setting up the Testo Framework with installing the `testo-nn-server` service. This service provides objects detection on screenshots. This server can be located either on the same machine as `testo` interpterer, or on a separate machine.

## Installing the package

You can download all packages [here](/en/downloads)

### Debian and Ubuntu

Installing the package itself:

```bash
sudo dpkg -i testo-nn-server.deb
```

Check if the server is running

```bash
sudo service testo-nn-server status
```

### CentOS

Installing the package itself:

```bash
rpm -i testo-nn-server.rpm
```

Check if the server is running

```bash
sudo service testo-nn-server status
```

## Windows 10

Run the installation package and follow the instructions.

## Setting up testo-nn-server

All testo-nn-server settings are located here: `/etc/testo/nn_server.json`. You can find further information on this matter [here](/en/docs/getting_started/nn_service_conf)

# Installing testo interpreter

## Enabling CPU Virtualization feature in BIOS

Make sure that you have Intel VT feature (if you have an Intel CPU) or AMD-V feature (if you have an AMD CPU) enabled in BIOS. Testo won't run without the CPU virtualization feature enabled.

## Installing prerequisites and packages

### Windows 10

> Testo Framework for Hyper-V works in experimental mode. Some actions and features are not available. Please read the docs thoroughly: all the unavailable for Hyper-V Testo features are labeled accordingly.

1. [Install](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v) the Hyper-V hypervisor.

2. [Download](/en/downloads) and launch the Testo Framework installation file. Then follow the instructions.

3. Open the command shell (cmd) as an administrator and run the command

``` bash
testo version
```

If the command returned a readable current Testo version - then Testo Framework had been installed successfully.

### Debian and Ubuntu

1. Installing Prerequisites:
```bash
sudo apt install libvirt0 libvirt-clients libvirt-daemon-system libguestfs0 qemu qemu-kvm ebtables dnsmasq-base
```
2. Installing the Testo package:
```bash
sudo dpkg -i testo-<version>.deb
```
3. Check the installation went ok:
```bash
testo version
```

It is also recommended (though not necessary) to install the package `virt-manager` - a GUI client for QEMU/KVM hypervisor. With virt-manager you can much easier observe the test runs, as well as control virtual machines manually when necessary. You can install the `virt-manager` with the command:

```bash
sudo apt install virt-manager
```

### CentOS

1.  Installing Prerequisites:
```bash
sudo yum -y install qemu-kvm libvirt libguestfs iptables-ebtables dnsmasq
```
2. Packages updating:
```bash
sudo yum update
```

After the updating a restart is required.

3. Installing the Testo package:
```bash
sudo rpm -i testo-<version>.rpm
```
4. Check the installation is ok
```bash
testo version
```

It is also recommended (though not necessary) to install the package `virt-manager` - a GUI client for QEMU/KVM hypervisor. With virt-manager you can much easier observe the test runs, as well as control virtual machines manually when necessary. You can install the `virt-manager` with the command:

```bash
sudo apt install virt-manager
```

> Testo Framework depends on the `libguestfs` library when managing virtual flash drives. Unfortunately, the default CentOS package for this library does not allow to use [virtual flash drives](/en/docs/lang/flash) with the NTFS file system. There is a workaround for this problem, involving compiling `libguestfs` from the [source code](https://www.redhat.com/archives/libguestfs/2016-February/msg00145.html).

## Setting up Testo-lang syntax highlighting

There is a Sublime Text 3 syntax highlighting plugin available. You can install this plugin with two possible ways.

### Sublime Text 3 Package Control

1. Install the [Package Control](https://packagecontrol.io/installation).
2. Open the command palette.
3. Select `Package Control: Install Package`.
4. Find and select package `Testo Highlighter`.

### Manual installation

1. Download the [file](https://github.com/testo-lang/testo-sublime/blob/master/Testo.sublime-syntax) with the Testo-lang syntax highlighting.
2. Copy this file to the `~/.config/sublime-text-3/Packages/User` folder.
3. Restart the Sublime Text 3.
