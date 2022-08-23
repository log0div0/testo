# Virtual Machine declaration

Vitual machine declaration starts with a `machine` keyword. The declaration itself has the following syntax:

```text
machine <name> {
	<attr1> [attr1_name]: <value1>
	<attr2> [attr2_name]: <value2>
	<attr3> [attr3_name]: <value3>
	...
}
```

> The declaration itself does not mean the actual creation of the virtual machine. The actual creation happens when the first test mentioning this virtual machine is run.

> Virtual machines can also be defined inside macros. See [here](Macros.md#macros-with-declarations) for more information.

Basically, a virtual machine declaration is a set of configuratoin attributes, some of which are mandatory. Attributes must be separated by newlines. At the end of the declaration another newline has to be placed.

For each virtual machine a `<name>` must be specified in the form of an [identifier](Language%20lexical%20elements.md#identifiers) or a string. If a string is used, the value inside the string must be convertible to an identifier. The virtual machine's name must be unique for all the virtual resoruces' names (e.g. virtual machines, virtual flash drives and virtual networks).

An attribute consists of an attribute's name, instance's name (only for several attributes) and a value (mandatory). Attribute names and instances' names are identifiers. At the moment, the only attributes requiring names for the instances, are the `nic` and `disk` attributes. Values' types depend on the attribute and can be one of:

- Positive integer.
- Memory size literal.
- String.
- Boolean literal.
- Attribute block.

For a virtual machine there is a set of **mandatory** attributes:

- `cpus` - Type: positive number or string. Number of cores for a virtual processor. If a string is used, the value inside the string must be convertible to positive integer.
- `ram` - Type: memory size literal or string. The amount of RAM for the virtual machine. If a string is used, the value inside the string must be convertible to a memory size literal.
- at least one `disk` attribute - Type: attribute block. Requires an instance's name. A disk drive coniguration.

**Optional attributes**:

- `iso` - Type: string. Path to the iso-image to be plugged into the DVD-drive after the virtual machine creation. Could be unplugged afterwards with an `unplug dvd` action.
- One or more `nic` - Type: attribute block. NIC configuration. Requires an instance's name.
- Exactly one `video` attribute - Type: attribute block. Video device configuration. Requires an instance's name. **Unavailable for Hyper-V**.
- `loader` - Type: string. Path to a custom loader blob file. **Unavailable for Hyper-V**.
- `qemu_enable_usb3` - Type: boolean. Enables USB 3 controller for the virtual machine. When the value is `false`, USB 2 controller is enabled instead.  Default value: `true`. **Unavailable for Hyper-V**.

## Disks configuration

Virtual machine disks configuration is done with the `disk` attributes. For each disk in a virtual machines a `disk` attribute is required. To distiguish multiple disks between themselves, every `disk` attribute must have a unique (inside the virtual machine) name. The value must be a block of inner attributes. The exact attribute choice depend on what mode you want to configure the disk into.

> You can specify several disks for a virtual machine. Just remember to choose unique names for each disk.

The following attributes are common for both modes:

- `bus` - Type: string. The name of bus the disk should be attached to. Possible values: `IDE` (default for amd64, not available on arm64) and `SCSI`.

### Blank disk creation mode

In this mode a new blank disk is created for the virtual machine. To activate this mode you have to specify the `size` attribute in the inner block of the `disk` attribute. The `size` attribute must be a **memory size literal** or a **string**. If a string is used, the value inside the string must be convertible to a memory size literal.

### Impoting an existing disk image

In this mode a copy of some **existing** disk image will be created for the virtual machine. With this mode you can manually prepare a virtual machine and then import your setups in the test scripts. To activate this mode you have to specifiy the `source` attribute in the inner block of the `disk` attribute. The `source` attribute must be a **string**. The source attribute must contain a path to the disk image you want to import.

> At the moment only `qcow2` disk images for QEMU and `vhdx` disk images for Hyper-V are supported.

> The original disk image will stay intact. When importing a disk image a copy is created (on arm64 the original disk is used as a backing file).

> If the original disk image belong to a virtual machine, this machine must be powered off when test scripts run.

> Attributes `size` and `source` are mutually exclusive. When one is used the other becomes disabled automatically.

> IMPORTANT NOTE! Please, do NOT install testo-guest-additions manually while preparing a virtual machine template! It could lead to unexpected results. Please, istall testo-guest-additions during tests execution only!

## NICs configuration

Network Interface Cards (NICs) configuration is done with the `nic` attributes. For each NIC for a virtual machine a `nic` attribute is required. Each NIC can be attached either to a network (subattribute `attached_to`) or to a Host NIC in the bridge mode (subattribute `attached_to_dev`). To distinguish multiple NICs between themselves every `nic` attribute must have a unique (inside the virtual machine) name. The value must be a block of attributes:

**Mandatory** `nic` attributes:

**ONE** of the following attributes:

- `attached_to` - Type: string. Network name to attach the NIC to. The network must be previously declared with the `network` directive. Can't be used with the `attached_to_dev` attribute.
- `attached_to_dev` - Type: string. Name of the Host NIC to attach the virtual NIC to. Can't be used with the `attached_to` attribute. **Not available for Hyper-V**.

**Optional** `nic` attributes:

- `adapter_type` - Type: string. A model for the NIC. Different NIC models are run by different drivers. Possible values on Linux: `ne2k_pci`, `i82551`, `i82557b`, `i82559er`, `rtl8139`, `e1000`, `pcnet`, `virtio`, `sungem`. If not specified, the hypervisor-default model will be selected. For QEMU the default value is `rtl8139`.  **Unavailable for Hyper-V**.
- `mac` - Type: string. MAC-address for the NIC. The value must be a valid MAC address (like `00:11:22:33:44:55`). If not specified, a random MAC-address will be generated for the NIC.

> Virtual machines are created with all the NICs attached and the links plugged into them. You can detach the NIC with the `unplug nic` action and unplug the virtual link from the NIC with the `unplug link` action.

## Video device configuration

> Unavailable for Hyper-V.

Virtual machines are always created with exactly one video device. By default, the model for this devies is picked based on the current hypervisor capabilities. In the most cases, the `vmvga` model is picked. If, for some reason, you want to choose some other model, you can specify it explicitly.

To do so you should use the `video` attribute instance of the virtual machine configuration. The attribute's instance must have a name.

> At the moment it is possible to use only one instance of the `video` attribute.

> The `video` attribute is optional, there is no need to specify it all the time. If the attribute is absent, the default video device is created.

`video` attribute takes a block of inner attributes as a value. There is only one inner attribute available at the moment:

- `adapter_type` - Type: string. Video device model. Possible values: `vmvga`, `qxl`, `cirrus`, `virtio`.

## Bootloader configuration

> Unavailable for Hyper-V.

By default, virtual machines use the SeaBIOS bootloader. If you want to use some custom bootloader (for example, UEFI-compatible), you should specify the path to this custom loader in the `loader` attribute.

For example, you can enable UEFI in a virtual machine with following steps:
1. Download [OVMF](https://wiki.ubuntu.com/UEFI/OVMF).
2. Specify the path to the `OVMF_CODE.fd` in the `loader` attribute (see example below).

## Shared folders

> Unavailable for Hyper-V

> Requires the Testo guest additions to be installed on the virtual machine

Shared folders is yet another way for virtual machines to interact with the Host (aside from virtual flash drives and the guest additions). To use shared folders you have to do two steps:

### 1. Add `shared_folder` attribute block

To enable shared folders it is required to specify one or more `shared_folder` attribute blocks in the virtual machine declaration:

```testo
shared_folder my_folder: {
	host_path: "/opt/shared_folder"
	readoly: false
}
```

Keep in mind that this attribute block must be named (all shared_folders names must be unique inside the virtual machine). There are also several nested subattributes:

**Mandatory `shared_folder` attributes**

- `host_path` - Type: string. A path to the shared folder on the Host side. **The folder must exist**.

**Optional `shared_folder` attributes**

- `readonly` - Type: boolean. Specifies whether the `readonly` mode must be enabled for the shared folder. With `readonly` enabled the virtual machine wouldn't be able to write data to this folder during test running. Default value: `false`.

### 2. Mount the folder inside the VM

You have to enable the shared folder inside the virtual machine as well. To do this you need to mount the folder with the following guest-additions command (this command should be executed inside the VM):

```bash
testo-guest-additions-cli mount <folder_name> <path_to_folder_on_guest> [--permanent]
```

Where
- `folder_name` - the name of the shared folder as presented in the virtual machine configuration (i.e. `my_folder` from the example above);
- `path_to_folder_on_guest` - the path to the shared folder inside the virtual machine;
- `permanent` - specify this if you want your shared folder to be mounted after a virtual machine reboot.

After this step the folder becomes available for data transferring between the Host and the VM.

> A shared folder is unmounted automatically before the snapshot creation (at the end of a test), and then mounted back automatically. The goal of that is to bypass a nasty bug in QEMU which occurs when a snapshot of a virtual machine with a shared folder mounted is created.

> You should make the shared folder available for read/write for the `qemu` process on the Host side. One way to achieve that is to add the line `user = "your_user"` at the end of the `/etc/libvirt/qemu.conf`, where `your_user` is the user owning the shared folder (on the Host). Don't forget to reboot the Host after that.

## Complete example

> This example is valid only for QEMU. For other hypervisors some attributes from the example below are unavailable.

Below you can see a complete example of a virtual machine configuration. Here are some main features:

1. When the virtual machine is created, the Ubuntu Server 16.04 iso-image is loaded into the virtual DVD-drive.
2. Two disks are created for the virtual machine: the `main` is copied (imported) from the existing disk image (from the manually created and prepared virtual machine `my_hand_mand_vm`) and the `secondary` is created empty, with the size specified in the `size_amount` param.
3. The virtual machine has 3 NICs: the `nat` will be used to connect the VM with the Internet and the `WAN` and the `LAN` will be used for isolated local area networks (`net1` and `net2`), presumably connecting the VM with other VMs.

Take notice that the `iso`, disk main's `source` and disk secondary's `size` attributes' values are calculated based on the `ISO_DIR`, `VM_DISK_POOL_DIR` and `size_amount` params respectively. If any of these params is not defined an error will be generated. For the `size` attribute an additional rule takes place: `size_amount` param value must be convertable to a memory size literal (for example, "2Gb"). Otherwise an errow will be generated.

```testo
machine example_machine {
	cpus: 1
	ram: 1024Mb
	iso: "${ISO_DIR}/ubuntu-16.04.6-server-amd64.iso"

	disk main: {
		source: "${VM_DISK_POOL_DIR}/my_hand_made_vm.qcow2"
	}

	disk secondary: {
		bus: "SCSI"
		size: "${size_amount}"
	}

	nic nat: {
		attached_to: "nat"
		adapter_type: "e1000"
	}

	nic WAN: {
		attached_to: "net2"
		mac: "52:54:00:00:00:00"
		adapter_type: "e1000"
	}

	nic LAN: {
		attached_to: "net1"
		mac: "52:54:00:00:00:11"
		adapter_type: "e1000"
	}

	nic HostNIC: {
		attached_to: "eth0" # The Host must have a NIC named "eth0"
		adapter_type: "e1000"
	}

	video main: {
		adapter_type: "qxl"
	}

	shared_folder my_folder: {
		host_path: "/opt/shared_folder"
		readoly: false
	}

	loader: "/usr/share/OVMF/OVMF_CODE.fd"

	qemu_enable_usb3: false
}
```

## Virtual machines caching

There is a cachine mechanism for virtual machines in the Testo Framework. This helps to check the integrity of VMs' configurations. If a configuration has changed since the last Testo run (and therefore the cache is lost), then the virtual machine must be re-created, and all the tests involving this virtual machine must be re-run. This is one of the checks performed when [evaluating](Tests.md#validating-the-test-cache) the cache integrity.

The complete virtual machines cache consistency checklist is this:

- Has any attribute changed since the last successful test run?
- Has the iso-image from the `iso` attribute changed since the last successful test run?
- Has the disk-image from the `source` attribute for the imported disks changed since the last successful test run?
- Has any `network` mentioned in `attached_to` in `nic` sections changed?
- Has the `loader` file changed?

If the answer to any of these questions is "yes", then the cache is lost and the virtual machine must be re-created.

> There are two ways to check the files' cache consistency for the `iso` and `source` attributes. If the file's size is less then 1 MB (which is basically impossible) then the file's integrity is evaluated based on its contents. Othwervise the evaluation is based on the Last modified timestamp of the file. You can adust the threshold of changing the evaluating mode with the `content_cksum_maxsize` command line argument.

## Manual virtual machine setup

The main Testo concept is to document every action happening with virtual machines right from the bare scratch - starting with the virtual machine clean and empty state (just what you get when you manually create a new virtual machine without importing pre-setup disks). This approach allows Testo to deploy required Test Benches with only `.testo` script files (which are easy to store and manage) and a few extra iso-files for the OS installation. This way all the Test Bench setup is placed right in front of your eyes (in the form of test scripts) and you can easily understand and change all that is done with the Test Bench.

However, this approach, of course, has a downside: scripting all the actions from the very beginning could be tedious and uneccessary in some cases. So to save you some labor, Testo-lang has a way to import a manually prepared virtual machine's disk into your Testo Test Bench. To import an existing virtual machine you should stick to the following algorithm:

1. Manually create a virtual machine to your liking. When manually creating the VM, you can choose any convenient configuration, but keep in mind the eventual Testo virtual machine configuration.
2. Manually set up the virtual machine to your liking (install OS, 3rd party software, change all the required settings and so on).
3. Turn off your VM.
4. Declare a new virtual machine in a `.testo` file and import the disk image from your manually created VM into it with the `source` attribute of a disk attribute.

Now your Testo-declared machine has a "clone disk" of your manually created VM and you can start the test scripts right from where you ended setting up your original VM.

Of course you could import any suitable disk image into a VM, not neccessary from another virtual machines. You could also combine various types of disks (newly created or imported) in the same virtual machine.

> The original VM disk's integrity is a part of cache consistency checks for Testo virtual machines. And therefore if you want to change the initial state of your imported VM you need just to power on your original VM, make some changes, power off the original VM and re-run the tests. The imported virtual machine's cache will be lost and all the according tests will be run again.

> You should consider the disk import as exactly **the disk only** import, not the whole machine import. There's no guarantee that Testo-created machine will have exactly the same hardware configuration as the original manually created virtual machine. In particular, the OS in the Testo-created machine may have other NICs naming order, than the OS in the original machine. Therefore it is **strongly not recommended** to setup the NICs (IP-addresses and so on) manually, because after the disk import the network settings may turn up lost. Instead, you should consider setting up the network in the test scripts.
