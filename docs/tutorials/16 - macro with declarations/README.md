# Tutorial 16. Macros with declarations

## What you're going to learn

In this tutorial you're going to learn about the last type of macros available in Testo-lang: macros with declarations.

## Introduction

Macros play a big role on Testo-lang. They allow you to put similar snippets of code into named blocks which you can reference later without the need to worry about their actual contents. In the 9th tutorial we've already got accustomed to the macros with actions and macros with commands. With macros we reduced the amount of copy-pasta in our code dramatically.

But there is one more type of macros which you need to learn to use the Test-lang capabilities to their fullest - macros with declarations. With this type of macros you can group up a bunch of tests or even whole tests benches!

## What to begin with?

To grasp the concept of macros with declarations, we should go back to the 9th part of our tutorials (the tutorial which introduces macros). Let's  take a look at our script we'd got at the end of the tutorial.

`declarations.testo`:

```testo
network internet {
	mode: "nat"
}

network LAN {
	mode: "internal"
}

param guest_additions_pkg "testo-guest-additions*"
param default_password "1111"

param client_hostname "client"
param client_login "client-login"

machine client {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}

	nic server_side: {
		attached_to: "LAN"
		mac: "52:54:00:00:00:AA"
	}
}

param server_hostname "server"
param server_login "server-login"

machine server {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}

	nic client_side: {
		attached_to: "LAN"
		mac: "52:54:00:00:00:BB"
	}
}

flash exchange_flash {
	fs: "ntfs"
	size: 8Mb
	folder: "./folder_to_copy"
}
```

`macros.testo`:

```testo
include "declarations.testo"

macro install_ubuntu(hostname, login, password = "${default_password}") {
	...
}


macro install_guest_additions(hostname, login, password="${default_password}") {
	...
}

macro unplug_nic(hostname, login, nic_name, password="${default_password}") {
	...
}

# A macro encapsulating several actions:
# 1) Plugging a flash drive
# 2) Mounting the flash drive into the filesystem
# 3) Executing a bash script
# 4) Unmounting the flash drive from the filesystem
# 5) Unplugging the flash drive
macro process_flash(flash_name, command) {
	plug flash "${flash_name}"
	sleep 5s
	exec bash "mount /dev/sdb1 /media"
	exec bash "${command}"
	exec bash "umount /media"
	unplug flash "${flash_name}"
}

# A command macro suitable for copying a file between arbitrary vms
# Copies a file from vm to /media/filename on the flash
# Copies a file from the flash to another vm
macro copy_file_with_flash(vm_from, vm_to, copy_flash, file_from, file_to) {
	"${vm_from}" process_flash("${copy_flash}", "cp ${file_from} /media/$(basename ${file_from})")
	"${vm_to}" process_flash("${copy_flash}", "cp /media/$(basename ${file_from}) ${file_to}")
}
```

`tests.testo`:

```testo
include "macros.testo"

test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}

test server_install_guest_additions: server_install_ubuntu {
	server install_guest_additions("${server_hostname}", "${server_login}")
}

test server_unplug_nat: server_install_guest_additions {
	server unplug_nic("${server_hostname}", "${server_login}", "nat")
}

test server_prepare: server_unplug_nat {
	server {
		copyto "./rename_net.sh" "/opt/rename_net.sh"
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:bb client_side
			ip a a 192.168.1.1/24 dev client_side
			ip l s client_side up
			ip ad
		"""
	}
}

test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}")
}

test client_install_guest_additions: client_install_ubuntu {
	client install_guest_additions("${client_hostname}", "${client_login}")
}

test client_unplug_nat: client_install_guest_additions {
	client unplug_nic("${client_hostname}", "${client_login}", "nat")
}

test client_prepare: client_unplug_nat {
	client {
		process_flash("exchange_flash", "cp /media/rename_net.sh /opt/rename_net.sh")

		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}

test test_ping: client_prepare, server_prepare {
	client exec bash "ping 192.168.1.2 -c5"
	server exec bash "ping 192.168.1.1 -c5"
}

# Take a note, that the test is inherited not from
# the test_ping, but from the client_prepare and
# server_prepare tests.
# Which means that test_ping and exchange_files_with_flash
# both lay on the samy tests tree level
test exchange_files_with_flash: client_prepare, server_prepare {
	#Create a file to be transferred
	client exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
	copy_file_with_flash("client", "server", "exchange_flash", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
	server exec bash "cat /tmp/copy_me_to_server.txt"
}
```

Looks good and elegant, doesn't it? Indeed, we've come a long way in that tutorial: we managed to organize all the code into a good shape, minimizing the amount of copy-pasta.

But let's consider this: what if we need to check that our tests run OK not only with Ubuntu Server 16.04, but Ubuntu Server 20.04 as well? The current virtual machines have Ubuntu Server 16.04 onboard, so to test Ubuntu 20.04 we'd need another two VMs (don't forget to rename the old ones so we won't get confused). To complete the picture let's also imagine we want to connect the new VMs with a new set of networks. Finally, we'd like to use a new virtual usb stick to copy files:

`declarations.testo`:

```testo
param guest_additions_pkg "testo-guest-additions*"
param default_password "1111"

param client_hostname "client"
param client_login "client-login"

param server_hostname "server"
param server_login "server-login"

network internet_16 {
	mode: "nat"
}

network LAN_16 {
	mode: "internal"
}

machine client_16 {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server_16.iso"

	nic nat: {
		attached_to: "internet_16"
	}

	nic server_side: {
		attached_to: "LAN_16"
		mac: "52:54:00:00:00:AA"
	}
}

machine server_16 {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server_16.iso"

	nic nat: {
		attached_to: "internet_16"
	}

	nic client_side: {
		attached_to: "LAN_16"
		mac: "52:54:00:00:00:BB"
	}
}

flash exchange_flash_16 {
	fs: "ntfs"
	size: 8Mb
	folder: "./folder_to_copy"
}

network internet_20 {
	mode: "nat"
}

network LAN_20 {
	mode: "internal"
}

machine client_20 {
	cpus: 1
	ram: 1Gb
	disk main: {
		size: 8Gb
	}
	iso: "${ISO_DIR}/ubuntu_server_20.iso"

	nic nat: {
		attached_to: "internet_20"
	}

	nic server_side: {
		attached_to: "LAN_20"
		mac: "52:54:00:00:00:AA"
	}
}

machine server_20 {
	cpus: 1
	ram: 1Gb
	disk main: {
		size: 8Gb
	}
	iso: "${ISO_DIR}/ubuntu_server_20.iso"

	nic nat: {
		attached_to: "internet_20"
	}

	nic client_side: {
		attached_to: "LAN_20"
		mac: "52:54:00:00:00:BB"
	}
}

flash exchange_flash_20 {
	fs: "ntfs"
	size: 8Mb
	folder: "./folder_to_copy"
}
```

Keep in mind that we had to increase the RAM and disk memory size for Ubuntu 20 VMs. Ubuntu 20 won't work with the amount of memory specified for Ubuntu 16.

Ok, but what's up with the tests? How do we need to change them? Ideally we'd want to copy the tests for Ubuntu 16 and just change the VM names in them... But there're a couple of issues.

First, the OS installation steps for Ubuntu 16 and 20 differ quite a lot. Which means we need a new macro:

```testo
macro install_ubuntu_server_20(nat_interface, hostname, login, password = "${default_password}") {
	unplug nic "${nat_interface}"
	start
	if (check "Language" timeout 5s) {
		press Enter
		wait "Install Ubuntu Server"; press Enter
	}
	wait "Welcome!" timeout 5m; press Enter

	wait "Please select your keyboard layout"; press Enter
	wait "Configure at least one interface"; press Enter
	wait "Proxy address"; press Enter
	wait "Mirror address"; press Enter
	wait "Continue without updating"; press Enter
	wait "Use an entire disk"; press Down*5; wait js "return find_text().match('Done').match_background('green').size() > 0"; press Enter
	wait "FILE SYSTEM SUMMARY"; press Enter
	wait "Confirm destructive action"; press Down, Enter
	wait "Enter the username"; type "${login}"; press Tab
	type "${hostname}"; press Tab
	type "${login}"; press Tab
	type "${password}"; press Tab
	type "${password}"; press Enter*2
	wait "SSH Setup"; press Tab, Enter
	wait "Installing system"
	wait "Installation complete!" timeout 15m
	press Enter
	wait "Please remove the installation medium"
	unplug dvd; press Enter
	wait "${hostname} login" timeout 3m

	stop; plug nic "${nat_interface}"; start

	wait "GNU GRUB"; press Enter

	wait "${hostname} login" timeout 3m
	type "${login}"; press Enter
	wait "Password"; type "${password}"; press Enter

	wait "${login}@${hostname}"
}
```

Let's rename the old macro into `install_ubuntu_server_16` to keep things streamlined.

And now we're going to combine both installation macros into one:

```testo
macro install_ubuntu_server(version, nat_interface, hostname, login, password = "${default_password}") {
	if ("${version}" STREQUAL "16") {
		install_ubuntu_server_16("${hostname}", "${login}", "${password}")
	} else if ("${version}" STREQUAL "20") {
		install_ubuntu_server_20("${nat_interface}","${hostname}", "${login}", "${password}")
	} else {
		abort "Unknown version:  ${version}"
	}
}
```

So now there is just a single macro which chooses the OS installation steps based on the `version` argument. With that issue settled, we're moving on.

Second, the `rename_net.sh` wouldn't work with Ubuntu 20.04 because the newer Ubuntu doesn't have the `ipconfig` utility installed (by default). We can solve this issue with a new script `rename_net_20.sh` using the `ip` utility:

```bash
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
```

And again, we're going to rename the `rename_net.sh` script into `rename_net_16.sh` to keep things streamlined.

That is all - this 2 issues aside, the tests for Ubuntu 20 and Ubuntu 16 are prety much the same thing. So let's put everything together and we'll see the next verion of `tests.testo`:

```testo
include "macros.testo"

test server_16_install_ubuntu {
	server_16 install_ubuntu_server("16", "nat", "${server_hostname}", "${server_login}")
}

test server_16_install_guest_additions: server_16_install_ubuntu {
	server_16 install_guest_additions("${server_hostname}", "${server_login}")
}

test server_16_unplug_nat: server_16_install_guest_additions {
	server_16 unplug_nic("${server_hostname}", "${server_login}", "nat")
}

test server_16_prepare: server_16_unplug_nat {
	server_16 {
		copyto "./rename_net_16.sh" "/opt/rename_net.sh"
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:bb client_side
			ip a a 192.168.1.1/24 dev client_side
			ip l s client_side up
			ip ad
		"""
	}
}

test client_16_install_ubuntu {
	client_16 install_ubuntu_server("16", "nat", "${client_hostname}", "${client_login}")
}

test client_16_install_guest_additions: client_16_install_ubuntu {
	client_16 install_guest_additions("${client_hostname}", "${client_login}")
}

test client_16_unplug_nat: client_16_install_guest_additions {
	client_16 unplug_nic("${client_hostname}", "${client_login}", "nat")
}

test client_16_prepare: client_16_unplug_nat {
	client_16 {
		process_flash("exchange_flash_16", "cp /media/rename_net_16.sh /opt/rename_net.sh")

		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}

test test_ping_16: client_16_prepare, server_16_prepare {
	client_16 exec bash "ping 192.168.1.2 -c5"
	server_16 exec bash "ping 192.168.1.1 -c5"
}

# Take a note, that the test is inherited not from
# the test_ping, but from the client_prepare and
# server_prepare tests.
# Which means that test_ping and exchange_files_with_flash
# both lay on the samy tests tree level
test exchange_files_with_flash_16: client_16_prepare, server_16_prepare {
	#Create a file to be transferred
	client_16 exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
	copy_file_with_flash("client_16", "server_16", "exchange_flash_16", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
	server_16 exec bash "cat /tmp/copy_me_to_server.txt"
}


test server_20_install_ubuntu {
	server_20 install_ubuntu_server("20", "nat", "${server_hostname}", "${server_login}")
}

test server_20_install_guest_additions: server_20_install_ubuntu {
	server_20 install_guest_additions("${server_hostname}", "${server_login}")
	server_20 {
		exec bash """
			ip l s ens7 up
			dhclient ens7
			apt install -y net-tools
		"""
	}
}

test server_20_unplug_nat: server_20_install_guest_additions {
	server_20 unplug_nic("${server_hostname}", "${server_login}", "nat")
}

test server_20_prepare: server_20_unplug_nat {
	server_20 {
		copyto "./rename_net_20.sh" "/opt/rename_net.sh"
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:bb client_side
			ip a a 192.168.1.1/24 dev client_side
			ip l s client_side up
			ip ad
		"""
	}
}

test client_20_install_ubuntu {
	client_20 install_ubuntu_server("20", "nat", "${client_hostname}", "${client_login}")
}

test client_20_install_guest_additions: client_20_install_ubuntu {
	client_20 install_guest_additions("${client_hostname}", "${client_login}")
	client_20 {
		exec bash """
			ip l s ens7 up
			dhclient ens7
			apt install -y net-tools
		"""
	}
}

test client_20_unplug_nat: client_20_install_guest_additions {
	client_20 unplug_nic("${client_hostname}", "${client_login}", "nat")
}

test client_20_prepare: client_20_unplug_nat {
	client_20 {
		process_flash("exchange_flash_20", "cp /media/rename_net_20.sh /opt/rename_net.sh")

		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}

test test_ping_20: client_20_prepare, server_20_prepare {
	client_20 exec bash "ping 192.168.1.2 -c5"
	server_20 exec bash "ping 192.168.1.1 -c5"
}

# Take a note, that the test is inherited not from
# the test_ping, but from the client_prepare and
# server_prepare tests.
# Which means that test_ping and exchange_files_with_flash
# both lay on the samy tests tree level
test exchange_files_with_flash_20: client_20_prepare, server_20_prepare {
	#Create a file to be transferred
	client_20 exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
	copy_file_with_flash("client_20", "server_20", "exchange_flash_20", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
	server_20 exec bash "cat /tmp/copy_me_to_server.txt"
}
```

Let's make sure this works:

![](imgs/terminal1.svg)

All done! We're managed to do the job, but once again we're forced to do copy-paste: we virtually duplicated the code for virtual machines, networks, flash drives and the tests themselves.

But we can fix that with a new type of macros in Testo-lang: macros with declarations.

## Macro with test bench declarations

Macros with declarations let you grouo up some of the top-level statements in Testl-lang: virtual entities declarations (`machine`, `flash`, `network`) and tests declarations. However, It is **prohibited** to declare params and other macros inside the macro body. You also can't use the `include` directive inside a macro. But on the other hand, you can *call* other macros with declarations.

We'll start with grouping up the VM, flash drive and network declarations into a new macro `generate_bench`:

```testo
macro generate_bench(version, ram_size, disk_size) {
	network "internet_${version}" {
		mode: "nat"
	}

	network "LAN_${version}" {
		mode: "internal"
	}

	machine "client_${version}" {
		cpus: 1
		ram: "${ram_size}"
		disk main: {
			size: "${disk_size}"
		}
		iso: "${ISO_DIR}/ubuntu_server_${version}.iso"

		nic nat: {
			attached_to: "internet_${version}"
		}

		nic server_side: {
			attached_to: "LAN_${version}"
			mac: "52:54:00:00:00:AA"
		}
	}

	machine "server_${version}" {
		cpus: 1
		ram: "${ram_size}"
		disk main: {
			size: "${disk_size}"
		}
		iso: "${ISO_DIR}/ubuntu_server_${version}.iso"

		nic nat: {
			attached_to: "internet_${version}"
		}

		nic client_side: {
			attached_to: "LAN_${version}"
			mac: "52:54:00:00:00:BB"
		}
	}

	flash "exchange_flash_${version}" {
		fs: "ntfs"
		size: 8Mb
		folder: "./folder_to_copy"
	}
}

generate_bench("16", "512Mb", "5Gb")
generate_bench("20", "1Gb", "8Gb")
```

A couple of things worth noticing.

Inside the macro we incapsulated the virtual test bench declaration: two vritual machines, two networks and a flash drive. Each `generate_bench` macro call will generate a whole new test bench. To prevent the virtual entities from "overplapping" each other with several macro calls, we made all the entities' names parameterized. To do so, we've used the Testo-lang feature of declaring entities' names with strings instead of identifiers (as we've been doing pretty much all the time before). And inside the strings we can reference the macro arguments.

So the macro `generate_bench` call with the `version = "16"` (Ubuntu Server version 16) will generate entities `network internet_16`, `network LAN_16`, `machine client_16`, `machine server_16` and `flash exchange_flash_16`. To generate the entities for Ubuntu 20 we just need to call the macro `generate_bench` with the `version = "16"` instead. This will "instruct" the virtual machines to carry the `ubuntu_server_20.iso` ISO-image (where the Ubuntu Server 20.04 image should be placed) instead of `ubuntu_server_16.iso`.

The next interesting thing is adjustable RAM (`ram`) and disk (`size`) sizes. As mentioned above, Ubuntu Server 20 wouldn't requires more memory (RAM and disk) thatn Ubuntu 16.04. It would be nice to have the opportunity to specify the memory amount for different test bench versions. We're going to do just that with the `ram_size` and `disk_size` arguments.

After the arguments get inside the macro, we can reference them inside the `ram` and `size` attributes. We're used to specifiy the memory amount in attributes with literals (`512Mb`, `5Gb`) but in Testo-lang it is also possible to do that with strings (`"512Mb"`, `"5Gb"`). Using the strings has a benefit of its own: you can reference macro args inside them, which is exactly what we did.

At the same time, the macro arguments values must be convertible to a memory amount specifier. It you tried to pass a wrong value (for instance, "5"), an error would be generated.

We declared the `generate_bench` macro, but it doesn't mean we declared toe virtuam machines, networks and flash drive themselves. To do that, we need to actually *call* the macro. The calls are made in the end of the `declarations.testo` and after the calls two separate and independent sets of machines, networks and flash drives are generated. Tou should also notice that the calls take place at the top level - the same level where all the usual declarations take place.

## Macro with tests declarations

It's time to move on to the tests. Tests' declarations (as well as virtual machines' declarations) can be encapsulated insode a macro (it's also possible to combine both tests and virtual machines declarations inside the same macro). Let's try to do this:

`tests.testo`:

```testo
include "macros.testo"

macro generate_tests(version) {

	test "server_${version}_install_ubuntu" {
		"server_${version}" install_ubuntu_server("${version}", "nat", "${server_hostname}", "${server_login}")
	}

	test "server_${version}_install_guest_additions": "server_${version}_install_ubuntu" {
		"server_${version}" install_guest_additions("${server_hostname}", "${server_login}")
	}

	test "server_${version}_unplug_nat": "server_${version}_install_guest_additions" {
		"server_${version}" unplug_nic("${server_hostname}", "${server_login}", "nat")
	}

	test "server_${version}_prepare": "server_${version}_unplug_nat" {
		"server_${version}" {
			copyto "./rename_net_${version}.sh" "/opt/rename_net.sh"
			exec bash """
				chmod +x /opt/rename_net.sh
				/opt/rename_net.sh 52:54:00:00:00:bb client_side
				ip a a 192.168.1.1/24 dev client_side
				ip l s client_side up
				ip ad
			"""
		}
	}

	test "client_${version}_install_ubuntu" {
		"client_${version}" install_ubuntu_server("${version}", "nat", "${client_hostname}", "${client_login}")
	}

	test "client_${version}_install_guest_additions": "client_${version}_install_ubuntu" {
		"client_${version}" install_guest_additions("${client_hostname}", "${client_login}")
	}

	test "client_${version}_unplug_nat": "client_${version}_install_guest_additions" {
		"client_${version}" unplug_nic("${client_hostname}", "${client_login}", "nat")
	}

	test "client_${version}_prepare": "client_${version}_unplug_nat" {
		"client_${version}" {
			process_flash("exchange_flash_${version}", "cp /media/rename_net_${version}.sh /opt/rename_net.sh")

			exec bash """
				chmod +x /opt/rename_net.sh
				/opt/rename_net.sh 52:54:00:00:00:aa server_side
				ip a a 192.168.1.2/24 dev server_side
				ip l s server_side up
				ip ad
			"""
		}
	}

	test "test_ping_${version}": "client_${version}_prepare", "server_${version}_prepare" {
		"client_${version}" exec bash "ping 192.168.1.2 -c5"
		"server_${version}" exec bash "ping 192.168.1.1 -c5"
	}

	# Take a note, that the test is inherited not from
	# the test_ping, but from the client_prepare and
	# server_prepare tests.
	# Which means that test_ping and exchange_files_with_flash
	# both lay on the samy tests tree level
	test "exchange_files_with_flash_${version}": "client_${version}_prepare", "server_${version}_prepare" {
		#Create a file to be transferred
		"client_${version}" exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
		copy_file_with_flash("client_${version}", "server_${version}", "exchange_flash_${version}", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
		"server_${version}" exec bash "cat /tmp/copy_me_to_server.txt"
	}
}

generate_tests("16")
generate_tests("20")
```

The `generate_tests` macro generates a whole bunch of tests: from the OS installation to the ping probing. Thanks to this macro, we're able to generalize the tests for Ubuntu 16.04 and Ubuntu 20.04. Let's take a look at a couple of interesting things.

First, let's taka a look at the tests' names. Just like we did before for the virtual entities, we're going to specify the tests' names with strings rather thatn identifiers. With strings we can access the macro arguments. We're also going to exploit the feature of specifying the tests parents' names with strings.

Second, inside the `generate_tests` macro we can call other types of macros: macros with actions (`install_ubuntu_server`, for instance) and with commands (`copy_file_with_flash`).

Keep in mind, that the installation proccess differs for Ubuntu Server 16.04 and Ubuntu Server 20.04. But we can pay no attention to that in our tests: since we've combined both installation types in the single `install_ubuntu_server` macro, inside the `server_${version}_install_ubuntu` we may not care about the actual way the installation is done. This is "the problem" of the `install_ubuntu_server` macro.

To generate the tests themselves, we need to call the `generate_tests` macro. We're going to call it two times: one for generating the tests collection for Ubuntu Server 16.04 (`generate_tests("16")`) and one more time to generate the tests collection for Ubuntu Server 20.04 (`generate_tests("20")`).

After that we can try out out new script and check that everythings works as planned.

## Conclusions

With macros it is possible to group up not only actions and commands, but whole declarations (virtual test benches and tests) as well. If you need to generate a bunch of similar-looking tests, then macros with declaration are really worth the considerations to do just that.

Congratulations! You've learned all the Testo-lang basics and may now proceed to creating your own test scripts!