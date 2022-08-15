
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const snippet1 =
`
network internet {
	mode: "nat"
}

machine server {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "\${ISO_DIR}\\ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}
}

param server_hostname "server"
param server_login "server-login"
param default_password "1111"

test server_install_ubuntu {
	server {
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
		wait "Hostname:" timeout 5m; press Backspace*36; type "\${server_hostname}"; press Enter
		wait "Full name for the new user"; type "\${server_login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "\${default_password}"; press Enter
		wait "Re-enter password to verify"; type "\${default_password}"; press Enter
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
		wait "login:" timeout 2m; type "\${server_login}"; press Enter
		wait "Password:"; type "\${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test server_prepare: server_install_ubuntu {
	server {
		# Enter sudo mode
		type "sudo su"; press Enter
		wait "password for \${server_login}"; type "\${default_password}"; press Enter
		wait "root@\${server_hostname}"

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

		wait "login:" timeout 2m; type "\${server_login}"; press Enter
		wait "Password:"; type "\${default_password}"; press Enter
		wait "Welcome to Ubuntu"

		# Enter sudo once more
		type "sudo su"; press Enter;
		wait "password for \${server_login}"; type "\${default_password}"; press Enter
		wait "root@\${server_hostname}"

		# Load the hv_sock module
		type "clear && modprobe hv_sock && echo Result is $?"; press Enter;
		wait "Result is 0"

		type "clear && lsmod | grep hv"; press Enter
		wait "hv_sock"
	}
}

param guest_additions_pkg "testo-guest-additions.deb"

test server_install_guest_additions: server_prepare {
	server {
		plug dvd "\${ISO_DIR}\\testo-guest-additions-hyperv.iso"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"
		type "clear && dpkg -i /media/\${guest_additions_pkg} && echo Result is $?"; press Enter;
		wait "Result is 0"
		type "clear && umount /media && echo Result is $?"; press Enter;
		wait "Result is 0"
		sleep 2s
		unplug dvd
	}
}
`

export const terminal1 = (
	<Terminal height="300px">
		<span className="">C:\Users\Testo&gt; testo clean<br/></span>
		<span className="">Testo is about to erase the following entities:<br/></span>
		<span className="">Virtual networks:<br/></span>
		<span className="">		- internet<br/></span>
		<span className="">Virtual machines:<br/></span>
		<span className="">		- my_ubuntu<br/></span>
		<span className=""><br/></span>
		<span className="">Do you confirm erasing these entities? [y/N]: y<br/></span>
		<span className="">Deleted network internet<br/></span>
		<span className="">Deleted virtual machine my_ubuntu<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span className="">C:\Users\Testo&gt; testo run ping.testo --stop_on_fail --param ISO_DIR C:\iso<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">server_install_ubuntu<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">server_install_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Install Ubuntu Server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Choose the language" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Select your location" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Detect keyboard layout?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Country of origin for the keyboard" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Keyboard layout" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Hostname:" </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">BACKSPACE </span>
		<span className="blue ">36 times </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"server" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Full name for the new user" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"server-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Username for your account" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Choose a password for the new user" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Re-enter password to verify" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Use weak password?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Encrypt your home directory?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Is this time zone correct?" </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Partitioning method" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Select disk to partition" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Write the changes to disks and configure LVM?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Amount of volume group to use for guided partitioning" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Force UEFI installation?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Write the changes to disks?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"HTTP proxy information" </span>
		<span className="blue ">for 3m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"How do you want to manage upgrades" </span>
		<span className="blue ">for 6m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Choose software to install" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Installation complete" </span>
		<span className="blue ">for 30m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Unplugging dvd </span>
		<span className="yellow "> </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"login:" </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"server-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Password:" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Welcome to Ubuntu" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">server_install_ubuntu</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="green bold">[ 33%] Test </span>
		<span className="yellow bold">server_install_ubuntu</span>
		<span className="green bold"> PASSED in 0h:7m:6s<br/></span>
		<span className="blue ">[ 33%] Preparing the environment for test </span>
		<span className="yellow ">server_prepare<br/></span>
		<span className="blue ">[ 33%] Running test </span>
		<span className="yellow ">server_prepare<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"sudo su" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"password for server-login" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"root@server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"dhclient -r eth0 && dhclient eth0 && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"clear && apt update && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"Result is 0" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"clear && apt install -y linux-azure && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"Result is 0" </span>
		<span className="blue ">for 15m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"reboot" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"login:" </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"server-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"Password:" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"Welcome to Ubuntu" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"sudo su" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"password for server-login" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"root@server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"clear && modprobe hv_sock && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"Result is 0" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"clear && lsmod | grep hv" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">"hv_sock" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 33%] Taking snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="green bold">[ 67%] Test </span>
		<span className="yellow bold">server_prepare</span>
		<span className="green bold"> PASSED in 0h:5m:45s<br/></span>
		<span className="blue ">[ 67%] Preparing the environment for test </span>
		<span className="yellow ">server_install_guest_additions<br/></span>
		<span className="blue ">[ 67%] Running test </span>
		<span className="yellow ">server_install_guest_additions<br/></span>
		<span className="blue ">[ 67%] Plugging dvd </span>
		<span className="yellow ">C:/iso/testo-guest-additions-hyperv.iso </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"mount /dev/cdrom /media" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"mounting read-only" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"clear && dpkg -i /media/testo-guest-additions.deb && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"Result is 0" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"clear && umount /media && echo Result is $?" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"Result is 0" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 2s<br/></span>
		<span className="blue ">[ 67%] Unplugging dvd </span>
		<span className="yellow "> </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Taking snapshot </span>
		<span className="yellow ">server_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">server_install_guest_additions</span>
		<span className="green bold"> PASSED in 0h:0m:15s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:13m:7s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="700px">
		<span className="">C:\Users\Testo&gt; testo run ping.testo --stop_on_fail --param ISO_DIR C:\iso<br/></span>
		<span className="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span className="">	- server_install_ubuntu<br/></span>
		<span className="">	- server_prepare<br/></span>
		<span className="">	- server_install_guest_additions<br/></span>
		<span className="">Do you confirm running them? [y/N]: y<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">server_install_ubuntu<br/></span>
		<span className="blue ">[  0%] Creating virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">server_install_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Install Ubuntu Server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Choose the language" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Select your location" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Detect keyboard layout?" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Country of origin for the keyboard" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Keyboard layout" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Hostname:" </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="red bold">C:/Users/Testo/ping.testo:63:3: Error while performing action wait "Hostname:" timeout 5m on virtual machine server<br/>  - Timeout<br/><br/>C:/Users/Testo/ng.testo:29:1: note: the virtual machine server was declared here<br/><br/></span>
		<span className="red bold">[ 17%] Test </span>
		<span className="yellow bold">server_install_ubuntu</span>
		<span className="red bold"> FAILED in 0h:5m:14s<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="600px">
		<span className="">C:\Users\Testo&gt; testo run ping.testo --stop_on_fail --param ISO_DIR C:\iso --test_spec server_setup_nic<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">server_setup_nic<br/></span>
		<span className="blue ">[ 75%] Preparing the environment for test </span>
		<span className="yellow ">server_setup_nic<br/></span>
		<span className="blue ">[ 75%] Restoring snapshot </span>
		<span className="yellow ">server_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 75%] Running test </span>
		<span className="yellow ">server_setup_nic<br/></span>
		<span className="blue ">[ 75%] Copying </span>
		<span className="yellow ">C:/Users/Testo/rename_net.sh </span>
		<span className="blue ">to virtual machine </span>
		<span className="yellow ">server </span>
		<span className="blue ">to destination </span>
		<span className="yellow ">/opt/rename_net.sh </span>
		<span className="blue ">with timeout 10m<br/></span>
		<span className="blue ">[ 75%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ chmod +x /opt/rename_net.sh<br/>+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/>Renaming success<br/>+ ip ad<br/>1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000<br/>    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span className=" ">    inet 127.0.0.1/8 scope host lo<br/>       valid_lft forever preferred_lft forever<br/>    inet6 ::1/128 scope host <br/>       valid_lft forever preferred_lft forever<br/>2: eth0: &lt;NO-CARRIER,BROADCAST,MULTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000</span>
		<span className=" "><br/>    link/ether 00:15:5d:01:25:15 brd ff:ff:ff:ff:ff:ff<br/>    inet 172.17.172.120/28 brd 172.17.172.127 scope global eth0<br/>       valid_lft forever preferred_lft forever<br/>    inet6 fe80::215:5dff:fe01:2515/64 scope link <br/>       valid_lft forever preferred_lft </span>
		<span className=" ">forever<br/>3: client_side: &lt;NO-CARRIER,BROADCAST,MULTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000<br/>    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/></span>
		<span className="blue ">[ 75%] Taking snapshot </span>
		<span className="yellow ">server_setup_nic</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">server_setup_nic</span>
		<span className="green bold"> PASSED in 0h:0m:4s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:4s<br/></span>
		<span className="blue bold">UP-TO-DATE: 3<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal5 = (
	<Terminal height="700px">
		<span className="">C:\Users\Testo&gt; testo run ping.testo --stop_on_fail --param ISO_DIR C:\iso<br/></span>
		<span className="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span className="">	- server_setup_nic<br/></span>
		<span className="">Do you confirm running them? [y/N]: y<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">server_setup_nic<br/></span>
		<span className="magenta ">client_setup_nic<br/></span>
		<span className="blue ">[ 75%] Preparing the environment for test </span>
		<span className="yellow ">server_setup_nic<br/></span>
		<span className="blue ">[ 75%] Restoring snapshot </span>
		<span className="yellow ">server_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 75%] Running test </span>
		<span className="yellow ">server_setup_nic<br/></span>
		<span className="blue ">[ 75%] Copying </span>
		<span className="yellow ">C:/Users/Testo/rename_net.sh </span>
		<span className="blue ">to virtual machine </span>
		<span className="yellow ">server </span>
		<span className="blue ">to destination </span>
		<span className="yellow ">/opt/rename_net.sh </span>
		<span className="blue ">with timeout 10m<br/></span>
		<span className="blue ">[ 75%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ chmod +x /opt/rename_net.sh<br/>+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/>Renaming success<br/>+ ip a a 192.168.1.1/24 dev client_side<br/>+ ip l s client_side up<br/>+ ip ad<br/>1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen</span>
		<span className=" "> 1000<br/>    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/>    inet 127.0.0.1/8 scope host lo<br/>       valid_lft forever preferred_lft forever<br/>    inet6 ::1/128 scope host <br/>       valid_lft forever preferred_lft forever<br/>2: eth0: &lt;NO-CARRIER,BROADCAST,MU</span>
		<span className=" ">LTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000<br/>    link/ether 00:15:5d:01:25:15 brd ff:ff:ff:ff:ff:ff<br/>    inet 172.17.172.120/28 brd 172.17.172.127 scope global eth0<br/>       valid_lft forever preferred_lft forever<br/>    inet6 fe80::215:5dff:</span>
		<span className=" ">fe01:2515/64 scope link <br/>       valid_lft forever preferred_lft forever<br/>3: client_side: &lt;NO-CARRIER,BROADCAST,MULTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000<br/>    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/>    inet 192.168.1.1/24 </span>
		<span className=" ">scope global client_side<br/>       valid_lft forever preferred_lft forever<br/></span>
		<span className="blue ">[ 75%] Taking snapshot </span>
		<span className="yellow ">server_setup_nic</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="green bold">[ 88%] Test </span>
		<span className="yellow bold">server_setup_nic</span>
		<span className="green bold"> PASSED in 0h:0m:5s<br/></span>
		<span className="blue ">[ 88%] Preparing the environment for test </span>
		<span className="yellow ">client_setup_nic<br/></span>
		<span className="blue ">[ 88%] Restoring snapshot </span>
		<span className="yellow ">client_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 88%] Running test </span>
		<span className="yellow ">client_setup_nic<br/></span>
		<span className="blue ">[ 88%] Copying </span>
		<span className="yellow ">C:/Users/Testo/rename_net.sh </span>
		<span className="blue ">to virtual machine </span>
		<span className="yellow ">client </span>
		<span className="blue ">to destination </span>
		<span className="yellow ">/opt/rename_net.sh </span>
		<span className="blue ">with timeout 10m<br/></span>
		<span className="blue ">[ 88%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ chmod +x /opt/rename_net.sh<br/>+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/>Renaming success<br/>+ ip a a 192.168.1.2/24 dev server_side<br/>+ ip l s server_side up<br/>+ ip ad<br/>1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen</span>
		<span className=" "> 1000<br/>    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/>    inet 127.0.0.1/8 scope host lo<br/>       valid_lft forever preferred_lft forever<br/>    inet6 ::1/128 scope host <br/>       valid_lft forever preferred_lft forever<br/>2: eth0: &lt;NO-CARRIER,BROADCAST,MU</span>
		<span className=" ">LTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000<br/>    link/ether 00:15:5d:01:25:16 brd ff:ff:ff:ff:ff:ff<br/>    inet 172.17.172.116/28 brd 172.17.172.127 scope global eth0<br/>       valid_lft forever preferred_lft forever<br/>    inet6 fe80::215:5dff:</span>
		<span className=" ">fe01:2516/64 scope link <br/>       valid_lft forever preferred_lft forever<br/>3: server_side: &lt;NO-CARRIER,BROADCAST,MULTICAST,UP&gt; mtu 1500 qdisc mq state DOWN group default qlen 1000<br/>    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/>    inet 192.168.1.2/24 </span>
		<span className=" ">scope global server_side<br/>       valid_lft forever preferred_lft forever<br/></span>
		<span className="blue ">[ 88%] Taking snapshot </span>
		<span className="yellow ">client_setup_nic</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">client_setup_nic</span>
		<span className="green bold"> PASSED in 0h:0m:5s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 8 TESTS IN 0h:0m:10s<br/></span>
		<span className="blue bold">UP-TO-DATE: 6<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal6 = (
	<Terminal height="600px">
		<span className="">C:\Users\Testo&gt; testo run ping.testo --stop_on_fail --param ISO_DIR C:\iso --test_spec test_ping<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_setup_nic<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_setup_nic<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">test_ping<br/></span>
		<span className="blue ">[ 89%] Preparing the environment for test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 89%] Restoring snapshot </span>
		<span className="yellow ">server_setup_nic</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 89%] Restoring snapshot </span>
		<span className="yellow ">client_setup_nic</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 89%] Running test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.2 -c5<br/>PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/>64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.023 ms<br/>64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.032 ms<br/>64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.027 ms<br/>64 </span>
		<span className=" ">bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.029 ms<br/>64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.025 ms<br/><br/>--- 192.168.1.2 ping statistics ---<br/>5 packets transmitted, 5 received, 0% packet loss, time 4093ms<br/>rtt min/avg/max/mdev = 0.023/0.027/0.032/</span>
		<span className=" ">0.004 ms<br/></span>
		<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.1 -c5<br/>PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/>64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.015 ms<br/>64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.028 ms<br/>64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.024 ms<br/>64 </span>
		<span className=" ">bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.031 ms<br/>64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.030 ms<br/><br/>--- 192.168.1.1 ping statistics ---<br/>5 packets transmitted, 5 received, 0% packet loss, time 4085ms<br/>rtt min/avg/max/mdev = 0.015/0.025/0.031/</span>
		<span className=" ">0.008 ms<br/></span>
		<span className="blue ">[ 89%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 89%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">test_ping</span>
		<span className="green bold"> PASSED in 0h:0m:17s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 9 TESTS IN 0h:0m:17s<br/></span>
		<span className="blue bold">UP-TO-DATE: 8<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)
