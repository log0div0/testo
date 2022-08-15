# Guide 7. Linking up virtual machines

## What you're going to learn

In this guide you're going to learn:

1. How to operate multiple virtual machines in a single test.
2. Some additional `nic` subattributes.
3. More information about virtual networks.

## Preconditions

1. Testo Framework is installed.
2. Hyper-V is installed.
3. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `C:\iso\ubuntu_server.iso`. The location may be different, but in this case the `ISO_DIR` command-line param has to be adjusted accordingly.
4. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
5. The Host has the Internet access.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 6](06_caching) is complete.

## Introduction

In the third guide we learned about the virtual networks `nat` mode, which allows you to provide the Internet access to virtual machines. But, obviously, virtual networks could (and should) be used for linking up virtual machines with each other as well. That's what we're going to do in this guide. Additionally we'll discover a few little tricks to make our test scripts more convenient and easier to read.

At the end of this guide we're going to get the following test bench:

<img src="/static/docs/tutorials/qemu/07_ping/network.svg"/>

## What to begin with?

From this moment on, there will be two virtual machines in our test cases, playing the roles of a server and a client. To make our test scripts more transparent and easier to read, we will need to do some job:

1. Rename the virtual machine `my_ubuntu` into `server`;
2. Rename the params `hostname`, `login`, `password` into `server_hostname`, `server_login` and `default_password` respectively and adjust all the references to these params;
3. Rename the tests `ubuntu_installation`, `ubuntu_prepare` and `ubuntu_install_guest_additions` into `server_install_ubuntu`, `server_prepare` and `server_install_guest_additions` respectively;
4. Delete the test `ubuntu_guest_additions_demo`.

You should get the following script:

<Snippet id="snippet1"/>

Since we've renamed our virtual machine into `server`, Testo treats is as a brand new machine, therefore all the tests must run again, including the virtual machine creation.

But we need to keep in mind that the old virtual machine, `my_ubuntu`, **wouldn't go anywhere automatically**. Testo Framework "assumes" that this virtual machine may be of some use for you in the future, and thus shouldn't be removed. Still, we, as humans, understand that we won't need `my_ubuntu` anymore, so let's delete it:

<Asset id="terminal1"/>

You can see that the virtual network `internet` was deleted as well.

Now, before doing anything else, it's good to make sure we didn't break anything with our changes.

<Asset id="terminal2"/>

## Declaring the second virtual machine

Now it's time to declare our second virtual machine. As you've probably guessed, it's going to be named `client`, and will be mostly a copy of the `server` machine.

```testo
machine client {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}\\ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}
}
```

But if we leave our virtual machines as they are now, they won't be connected in any way. To link them up we need to declare a new virtual network:

```testo
network LAN {
	mode: "internal"
}
```

Take a note that this network is declared as `internal`, which means that it's designated to isolated connection between machines.

All that's left to do is to add new NICs to the virtual machines, with attachment to the `LAN` network:

```testo
machine client {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}\\ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}

	nic server_side: {
		attached_to: "LAN"
		mac: "52:54:00:00:00:AA"
	}
}

machine server {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}\\ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}

	nic client_side: {
		attached_to: "LAN"
		mac: "52:54:00:00:00:BB"
	}
}
```

You can see that we added the `mac` subattribute to the "internal" NICs. Knowing the exact mac will play its role a little later, when we'll do renaming NICs inside the OS (so we can distinguish them easily).

Of course, the new `client` machine needs a couple of preparatory tests of its own: `client_install_ubuntu`, `client_prepare` and `client_install_guest_additions`. Don't forget to add a couple of new params:

```testo
param client_hostname "client"
param client_login "client-login"

test client_install_ubuntu {
	client {
		start
		# The actions can be separated with a newline
		# or a semicolon
		wait "Install Ubuntu Server"; press Enter;
		wait "Choose the language";	press Enter
		wait "Select your location"; press Enter
		wait "Detect keyboard layout?";	press Enter
		wait "Country of origin for the keyboard"; press Enter
		wait "Keyboard layout"; press Enter
		#wait "No network interfaces detected" timeout 5m; press Enter
		wait "Hostname:" timeout 5m; press Backspace*36; type "${client_hostname}"; press Enter
		wait "Full name for the new user"; type "${client_login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "${default_password}"; press Enter
		wait "Re-enter password to verify"; type "${default_password}"; press Enter
		wait "Use weak password?"; press Left, Enter
		wait "Encrypt your home directory?"; press Enter

		wait "Is this time zone correct?" timeout 2m; press Enter
		wait "Partitioning method"; press Enter
		wait "Select disk to partition"; press Enter
		wait "Write the changes to disks and configure LVM?"; press Left, Enter
		wait "Amount of volume group to use for guided partitioning"; press Enter
		wait "Force UEFI installation?"; press Left, Enter
		wait "Write the changes to disks?"; press Left, Enter
		wait "HTTP proxy information" timeout 3m; press Enter
		wait "How do you want to manage upgrades" timeout 6m; press Enter
		wait "Choose software to install"; press Enter
		wait "Installation complete" timeout 30m;

		unplug dvd; press Enter
		wait "login:" timeout 2m; type "${client_login}"; press Enter
		wait "Password:"; type "${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test client_prepare: client_install_ubuntu {
	client {
		# Enter sudo mode
		type "sudo su"; press Enter
		wait "password for ${client_login}"; type "${default_password}"; press Enter
		wait "root@${client_hostname}"

		# Reset the eth0 NIC to prevent any issues with it after the rollback
		type "dhclient -r eth0 && dhclient eth0 && echo Result is $?"; press Enter

		# Check that apt is OK
		type "clear && apt update && echo Result is $?"; press Enter
		wait "Result is 0"

		# Install linux-azure package
		type "clear && apt install -y linux-azure && echo Result is $?"; press Enter
		wait "Result is 0" timeout 15m		

		# Reboot and login
		type "reboot"; press Enter

		wait "login:" timeout 2m; type "${client_login}"; press Enter
		wait "Password:"; type "${default_password}"; press Enter
		wait "Welcome to Ubuntu"

		# Enter sudo once more
		type "sudo su"; press Enter;
		wait "password for ${client_login}"; type "${default_password}"; press Enter
		wait "root@${client_hostname}"

		# Load the hv_sock module
		type "clear && modprobe hv_sock && echo Result is $?"; press Enter;
		wait "Result is 0"

		type "clear && lsmod | grep hv"; press Enter
		wait "hv_sock"
	}
}

test client_install_guest_additions: client_prepare {
	client {
		plug dvd "${ISO_DIR}\\testo-guest-additions-hyperv.iso"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"
		type "clear && dpkg -i /media/${guest_additions_pkg} && echo Result is $?"; press Enter;
		wait "Result is 0"
		type "clear && umount /media && echo Result is $?"; press Enter;
		wait "Result is 0"
		sleep 2s
		unplug dvd
	}
}
```

We can see a lot of duplicate code, but we shouldn't fret about that. Later we'll learn about macros, which allow to group up frequently-used actions into named blocks, and our sript will get much shorter.

Now let's run our script.

<Asset id="terminal3"/>

Look at that: Ubunstu installation broke up. Again. Why? Because now we have multiple NICs plugged into the virtual machine, so we get a new screen we haven't been expecting in the test script:

![Primary NIC](/static/docs/tutorials/hyperv/07_ping/primary_nic.png)

The primary interface is listed at the top, so we just need to press Enter. Let's adjust our test script a little.

```testo
wait "Keyboard layout"; press Enter
#wait "No network interfaces detected" timeout 5m; press Enter
wait "Primary network interface"; press Enter
wait "Hostname:" timeout 5m; press Backspace*36; type "${server_hostname}"; press Enter
```

Run the script again, and now all the tests should pass successfully.

## Renaming NICs inside the OS

By default, the NICs are given pretty uninformative names inside the OS (`ens3`, `ens4` and so on). Of course it would be much more convenient if we had them named accordingly to their names in the virtual machine declaration: `client_side`, `server side` and so on. We can actually achieve that with renaming NICs inside the OS based on the MAC, which we already know.

Renaming can be done with (for example), the following bash-script (download link is available at the end of the guide).

``` bash
#!/bin/bash

set -e

mac=$1

oldname=`ifconfig -a | grep ${mac,,} | awk '{print $1}'`
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

Save this script in the `rename_net.sh` file in the same folder where the `ping.testo` file is located.

And write a test to use this script inside the `server`:

```testo
test server_setup_nic: server_install_guest_additions {
	server {
		copyto ".\\rename_net.sh" "/opt/rename_net.sh"
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:bb client_side
			ip ad
		"""
	}
}
```

<Asset id="terminal4"/>

In the `ip` command output we can see that the remaining NIC is named clear and neat: `client_side`.

Now we need to setup the IP-address for the NIC:

```testo
test server_setup_nic: server_install_guest_additions {
	server {
		copyto ".\\rename_net.sh" "/opt/rename_net.sh"
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:bb client_side
			ip a a 192.168.1.1/24 dev client_side
			ip l s client_side up
			ip ad
		"""
	}
}
```

And repeat all these steps with the `client` virtual machine.

<Asset id="terminal5"/>

## Pinging!

Finally, all things are set up and we can check that `client` and `server` can ping each other.

Create one last test `test_ping` which, unlike the previous tests, has **two** parent-tests: `client_prepare` and `server_prepare`. We need both these tests to complete successfully to begin the ping testing.

```testo
test test_ping: client_setup_nic, server_setup_nic {
	client exec bash "ping 192.168.1.2 -c5"
	server exec bash "ping 192.168.1.1 -c5"
}
```

Final script run

<Asset id="terminal6"/>

We can see that the ping command runs great, which means that we managed to setup a test bench with two linked up virtual machines!

## Conclusions

Virtual networks can be used not only to gain the Internet access, but to link up virtual machines with each other as well.

Testo Framework allows you to develop preparatory tests with the Internet access and then unplug the NIC leading to the Internet if the Internet is not needed anymore. This way you can reduce the redundant Internet connections, that can possibly mess up the test cases.

You can simplify the NICs distibguishing inside the test scripts by assigning fixed MAC-addresses to the NICs and then renaming them inside the OS to your linking.

Tests hierarchy looks like this at the moment:

<img src="/static/docs/tutorials/hyperv/07_ping/test_hierarchy.svg"/>

You can find the complete test scripts and NIC-renaming bash script [here](https://github.com/testo-lang/testo-tutorials/tree/master/hyperv/07%20-%20ping).
