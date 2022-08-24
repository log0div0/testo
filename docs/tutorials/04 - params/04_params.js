
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const snippet1 =
`...

test ubuntu_installation {
	my_ubuntu {
		start
		...
		wait "Hostname:" timeout 30s; press Backspace*36; type "my-ubuntu"; press Enter
		wait "Full name for the new user"; type "my-ubuntu-login"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "1111"; press Enter
		wait "Re-enter password to verify"; type "1111"; press Enter
		...
		unplug dvd; press Enter
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"
		...
	}
}

...`

export const terminal1 = (
	<Terminal height="150px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
		<span class="">/home/alex/testo/hello_world.testo.testo:8:7: Error while resolving "$&#123;ISO_DIR&#125;/ubuntu_server.iso"<br/></span>
		<span class="">	-param "ISO_DIR" not defined<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="100px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation --param ISO_DIR /opt/iso<br/></span>
	</Terminal>
)

export const terminal3 = (
	<Terminal>
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">ubuntu_installation<br/></span>
		<span class="magenta ">guest_additions_installation<br/></span>
		<span class="magenta ">guest_additions_demo<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
		<span class="blue bold">UP-TO-DATE: 3<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso --invalidate \*<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">ubuntu_installation<br/></span>
		<span class="magenta ">guest_additions_installation<br/></span>
		<span class="magenta ">guest_additions_demo<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">ubuntu_installation<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">ubuntu_installation<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">English </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install Ubuntu Server </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose the language </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select your location </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Detect keyboard layout? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Country of origin for the keyboard </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Keyboard layout </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">No network interfaces detected </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 30s with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">BACKSPACE </span>
		<span class="blue ">36 times </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"my-ubuntu" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Full name for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"my-ubuntu-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Username for your account </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose a password for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Re-enter password to verify </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Use weak password? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Encrypt your home directory? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select your timezone </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Partitioning method </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select disk to partition </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks and configure LVM? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Amount of volume group to use for guided partitioning </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">HTTP proxy information </span>
		<span class="blue ">for 3m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">How do you want to manage upgrades </span>
		<span class="blue ">for 6m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose software to install </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install the GRUB boot loader to the master boot record? </span>
		<span class="blue ">for 10m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Installation complete </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Unplugging dvd </span>
		<span class="yellow "> </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">login: </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"my-ubuntu-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Password: </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Welcome to Ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">ubuntu_installation</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="green bold">[ 33%] Test </span>
		<span class="yellow bold">ubuntu_installation</span>
		<span class="green bold"> PASSED in 0h:4m:24s<br/></span>
		<span class="blue ">[ 33%] Preparing the environment for test </span>
		<span class="yellow ">guest_additions_installation<br/></span>
		<span class="blue ">[ 33%] Running test </span>
		<span class="yellow ">guest_additions_installation<br/></span>
		<span class="blue ">[ 33%] Plugging dvd </span>
		<span class="yellow ">/opt/iso/testo-guest-additions.iso </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Typing </span>
		<span class="yellow ">"sudo su" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Waiting </span>
		<span class="yellow ">password for my-ubuntu-login </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Waiting </span>
		<span class="yellow ">root@my-ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Typing </span>
		<span class="yellow ">"mount /dev/cdrom /media" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Waiting </span>
		<span class="yellow ">mounting read-only </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Typing </span>
		<span class="yellow ">"dpkg -i /media/*.deb" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Waiting </span>
		<span class="yellow ">Setting up testo-guest-additions </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Typing </span>
		<span class="yellow ">"umount /media" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Sleeping in virtual machine </span>
		<span class="yellow ">my_ubuntu</span>
		<span class="blue "> for 2s<br/></span>
		<span class="blue ">[ 33%] Unplugging dvd </span>
		<span class="yellow "> </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[ 33%] Taking snapshot </span>
		<span class="yellow ">guest_additions_installation</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="green bold">[ 67%] Test </span>
		<span class="yellow bold">guest_additions_installation</span>
		<span class="green bold"> PASSED in 0h:0m:14s<br/></span>
		<span class="blue ">[ 67%] Preparing the environment for test </span>
		<span class="yellow ">guest_additions_demo<br/></span>
		<span class="blue ">[ 67%] Running test </span>
		<span class="yellow ">guest_additions_demo<br/></span>
		<span class="blue ">[ 67%] Executing bash command in virtual machine </span>
		<span class="yellow ">my_ubuntu</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ echo Hello world<br/></span>
		<span class=" ">Hello world<br/></span>
		<span class=" ">+ echo from bash<br/></span>
		<span class=" ">from bash<br/></span>
		<span class="blue ">[ 67%] Executing python3 command in virtual machine </span>
		<span class="yellow ">my_ubuntu</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">Hello from python3!<br/></span>
		<span class="blue ">[ 67%] Taking snapshot </span>
		<span class="yellow ">guest_additions_demo</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">guest_additions_demo</span>
		<span class="green bold"> PASSED in 0h:0m:3s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:4m:42s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 3<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)
