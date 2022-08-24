
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const snippet1 =
`network internet {
	mode: "nat"
}

machine server {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "\${ISO_DIR}/ubuntu_server.iso"

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
		wait "English"
		press Enter
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

		#wait "Select your timezone" timeout 2m; press Enter
		wait "Is this time zone correct?" timeout 2m; press Enter
		wait "Partitioning method"; press Enter
		wait "Select disk to partition"; press Enter
		wait "Write the changes to disks and configure LVM?"; press Left, Enter
		wait "Amount of volume group to use for guided partitioning"; press Enter
		wait "Write the changes to disks?"; press Left, Enter
		wait "HTTP proxy information" timeout 3m; press Enter
		wait "How do you want to manage upgrades" timeout 6m; press Enter
		wait "Choose software to install"; press Enter
		wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
		wait "Installation complete" timeout 1m;

		unplug dvd; press Enter
		wait "server_login:" timeout 2m; type "\${server_login}"; press Enter
		wait "Password:"; type "\${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}

param guest_additions_pkg "testo-guest-additions*"
test server_install_guest_additions: server_install_ubuntu {
	server {
		plug dvd "\${ISO_DIR}/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		# Take a note that you may reference params in any part of the string
		wait "password for \${server_login}"; type "\${default_password}"; press Enter
		wait "root@\${server_hostname}"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"; type "dpkg -i /media/\${guest_additions_pkg}"; press Enter;
		wait "Setting up testo-guest-additions"
		type "umount /media"; press Enter;
		# Give a little time for the umount to do its job
		sleep 2s
		unplug dvd
	}
}`

export const terminal1 = (
	<Terminal height="300px">
		<span class="">user$ sudo testo clean<br/></span>
		<span class="">Testo is about to erase the following entities:<br/></span>
		<span class="">Virtual networks:<br/></span>
		<span class="">		- internet<br/></span>
		<span class="">Virtual machines:<br/></span>
		<span class="">		- my_ubuntu<br/></span>
		<span class=""><br/></span>
		<span class="">Do you confirm erasing these entities? [y/N]: y<br/></span>
		<span class="">Deleted network internet<br/></span>
		<span class="">Deleted virtual machine my_ubuntu<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">English </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install Ubuntu Server </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose the language </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select your location </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Detect keyboard layout? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Country of origin for the keyboard </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Keyboard layout </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">BACKSPACE </span>
		<span class="blue ">36 times </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"server" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Full name for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"server-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Username for your account </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose a password for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Re-enter password to verify </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Use weak password? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Encrypt your home directory? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Is this time zone correct? </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Partitioning method </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select disk to partition </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks and configure LVM? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Amount of volume group to use for guided partitioning </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">HTTP proxy information </span>
		<span class="blue ">for 3m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">How do you want to manage upgrades </span>
		<span class="blue ">for 6m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose software to install </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install the GRUB boot loader to the master boot record? </span>
		<span class="blue ">for 10m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Installation complete </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Unplugging dvd </span>
		<span class="yellow "> </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">server login: </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"server-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Password: </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Welcome to Ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">server_install_ubuntu</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[ 50%] Test </span>
		<span class="yellow bold">server_install_ubuntu</span>
		<span class="green bold"> PASSED in 0h:5m:35s<br/></span>
		<span class="blue ">[ 50%] Preparing the environment for test </span>
		<span class="yellow ">server_install_guest_additions<br/></span>
		<span class="blue ">[ 50%] Running test </span>
		<span class="yellow ">server_install_guest_additions<br/></span>
		<span class="blue ">[ 50%] Plugging dvd </span>
		<span class="yellow ">/opt/iso/testo-guest-additions.iso </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Typing </span>
		<span class="yellow ">"sudo su" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Waiting </span>
		<span class="yellow ">password for server-login </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Waiting </span>
		<span class="yellow ">root@server </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Typing </span>
		<span class="yellow ">"mount /dev/cdrom /media" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Waiting </span>
		<span class="yellow ">mounting read-only </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Typing </span>
		<span class="yellow ">"dpkg -i /media/testo-guest-additions*" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Waiting </span>
		<span class="yellow ">Setting up testo-guest-additions </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Typing </span>
		<span class="yellow ">"umount /media" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Sleeping in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> for 2s<br/></span>
		<span class="blue ">[ 50%] Unplugging dvd </span>
		<span class="yellow "> </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 50%] Taking snapshot </span>
		<span class="yellow ">server_install_guest_additions</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">server_install_guest_additions</span>
		<span class="green bold"> PASSED in 0h:0m:15s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 2 TESTS IN 0h:5m:51s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="700px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span class="">	- server_install_ubuntu<br/></span>
		<span class="">	- server_install_guest_additions<br/></span>
		<span class="">Do you confirm running them? [y/N]: y<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Creating virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">English </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install Ubuntu Server </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose the language </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select your location </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Detect keyboard layout? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Country of origin for the keyboard </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Keyboard layout </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="red bold">/home/alex/testo/hello_world.testo:61:3: Error while performing action wait Hostname: timeout 5m on virtual machine server:<br/>	-Timeout<br/></span>
		<span class="red bold">[ 25%] Test </span>
		<span class="yellow bold">server_install_ubuntu</span>
		<span class="red bold"> FAILED in 0h:5m:12s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="650px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_unplug_nat<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Preparing the environment for test </span>
		<span class="yellow ">server_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Restoring snapshot </span>
		<span class="yellow ">server_install_guest_additions</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Running test </span>
		<span class="yellow ">server_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Shutting down virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 1m<br/></span>
		<span class="blue ">[ 67%] Unplugging nic </span>
		<span class="yellow ">nat </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Starting virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">server login: </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"server-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">Password: </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">Welcome to Ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 67%] Taking snapshot </span>
		<span class="yellow ">server_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">server_unplug_nat</span>
		<span class="green bold"> PASSED in 0h:0m:28s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:28s<br/></span>
		<span class="blue bold">UP-TO-DATE: 2<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal5 = (
	<Terminal height="650px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec client_unplug_nat<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Preparing the environment for test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Restoring snapshot </span>
		<span class="yellow ">client_install_guest_additions</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Running test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Shutting down virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 1m<br/></span>
		<span class="blue ">[ 67%] Unplugging nic </span>
		<span class="yellow ">nat </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Starting virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">client login: </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"client-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">Password: </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">Welcome to Ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Taking snapshot </span>
		<span class="yellow ">client_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">client_unplug_nat</span>
		<span class="green bold"> PASSED in 0h:0m:28s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:28s<br/></span>
		<span class="blue bold">UP-TO-DATE: 2<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal6 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_prepare<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="blue ">[ 75%] Preparing the environment for test </span>
		<span class="yellow ">server_prepare<br/></span>
		<span class="blue ">[ 75%] Restoring snapshot </span>
		<span class="yellow ">server_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 75%] Running test </span>
		<span class="yellow ">server_prepare<br/></span>
		<span class="blue ">[ 75%] Copying </span>
		<span class="yellow ">./rename_net.sh </span>
		<span class="blue ">to virtual machine </span>
		<span class="yellow ">server </span>
		<span class="blue ">to destination </span>
		<span class="yellow ">/opt/rename_net.sh </span>
		<span class="blue ">with timeout 10m<br/></span>
		<span class="blue ">[ 75%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ chmod +x /opt/rename_net.sh<br/></span>
		<span class=" ">+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/></span>
		<span class=" ">Renaming success<br/></span>
		<span class=" ">+ ip ad<br/></span>
		<span class=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
		<span class=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span class=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 ::1/128 scope host <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">2: client_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
		<span class=" ">    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/></span>
		<span class=" ">    inet6 fe80::5054:ff:fe00:bb/64 scope link tentative <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class="blue ">[ 75%] Taking snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">server_prepare</span>
		<span class="green bold"> PASSED in 0h:0m:2s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:2s<br/></span>
		<span class="blue bold">UP-TO-DATE: 3<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal7 = (
	<Terminal height="700px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span class="">	- server_prepare<br/></span>
		<span class="">Do you confirm running them? [y/N]: y<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="blue ">[ 75%] Preparing the environment for test </span>
		<span class="yellow ">server_prepare<br/></span>
		<span class="blue ">[ 75%] Restoring snapshot </span>
		<span class="yellow ">server_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 75%] Running test </span>
		<span class="yellow ">server_prepare<br/></span>
		<span class="blue ">[ 75%] Copying </span>
		<span class="yellow ">./rename_net.sh </span>
		<span class="blue ">to virtual machine </span>
		<span class="yellow ">server </span>
		<span class="blue ">to destination </span>
		<span class="yellow ">/opt/rename_net.sh </span>
		<span class="blue ">with timeout 10m<br/></span>
		<span class="blue ">[ 75%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ chmod +x /opt/rename_net.sh<br/></span>
		<span class=" ">+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/></span>
		<span class=" ">Renaming success<br/></span>
		<span class=" ">+ ip a a 192.168.1.1/24 dev client_side<br/></span>
		<span class=" ">+ ip l s client_side up<br/></span>
		<span class=" ">+ ip ad<br/></span>
		<span class=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
		<span class=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span class=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 ::1/128 scope host <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">2: client_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
		<span class=" ">    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/></span>
		<span class=" ">    inet 192.168.1.1/24 scope global client_side<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 fe80::5054:ff:fe00:bb/64 scope link tentative <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class="blue ">[ 75%] Taking snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[ 88%] Test </span>
		<span class="yellow bold">server_prepare</span>
		<span class="green bold"> PASSED in 0h:0m:4s<br/></span>
		<span class="blue ">[ 88%] Preparing the environment for test </span>
		<span class="yellow ">client_prepare<br/></span>
		<span class="blue ">[ 88%] Restoring snapshot </span>
		<span class="yellow ">client_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 88%] Running test </span>
		<span class="yellow ">client_prepare<br/></span>
		<span class="blue ">[ 88%] Copying </span>
		<span class="yellow ">./rename_net.sh </span>
		<span class="blue ">to virtual machine </span>
		<span class="yellow ">client </span>
		<span class="blue ">to destination </span>
		<span class="yellow ">/opt/rename_net.sh </span>
		<span class="blue ">with timeout 10m<br/></span>
		<span class="blue ">[ 88%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ chmod +x /opt/rename_net.sh<br/></span>
		<span class=" ">+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/></span>
		<span class=" ">Renaming success<br/></span>
		<span class=" ">+ ip a a 192.168.1.2/24 dev server_side<br/></span>
		<span class=" ">+ ip l s server_side up<br/></span>
		<span class=" ">+ ip ad<br/></span>
		<span class=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
		<span class=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span class=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 ::1/128 scope host <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">2: server_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
		<span class=" ">    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/></span>
		<span class=" ">    inet 192.168.1.2/24 scope global server_side<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 fe80::5054:ff:fe00:aa/64 scope link tentative <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class="blue ">[ 88%] Taking snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">client_prepare</span>
		<span class="green bold"> PASSED in 0h:0m:4s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 8 TESTS IN 0h:0m:8s<br/></span>
		<span class="blue bold">UP-TO-DATE: 6<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal8 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec test_ping<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">test_ping<br/></span>
		<span class="blue ">[ 89%] Preparing the environment for test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 89%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 89%] Restoring snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 89%] Running test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 89%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.2 -c5<br/></span>
		<span class=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.056 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.036 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.046 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.043 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.2 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.036/0.045/0.056/0.008 ms<br/></span>
		<span class="blue ">[ 89%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.1 -c5<br/></span>
		<span class=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.038 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.042 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.043 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.042 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.1 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.038/0.044/0.057/0.008 ms<br/></span>
		<span class="blue ">[ 89%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 89%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">test_ping</span>
		<span class="green bold"> PASSED in 0h:0m:16s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 9 TESTS IN 0h:0m:16s<br/></span>
		<span class="blue bold">UP-TO-DATE: 8<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)
