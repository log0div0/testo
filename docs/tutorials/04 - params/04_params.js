
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
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
		<span className="">/home/alex/testo/hello_world.testo.testo:8:7: Error while resolving "$&#123;ISO_DIR&#125;/ubuntu_server.iso"<br/></span>
		<span className="">	-param "ISO_DIR" not defined<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="100px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation --param ISO_DIR /opt/iso<br/></span>
	</Terminal>
)

export const terminal3 = (
	<Terminal>
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">ubuntu_installation<br/></span>
		<span className="magenta ">guest_additions_installation<br/></span>
		<span className="magenta ">guest_additions_demo<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
		<span className="blue bold">UP-TO-DATE: 3<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso --invalidate \*<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">ubuntu_installation<br/></span>
		<span className="magenta ">guest_additions_installation<br/></span>
		<span className="magenta ">guest_additions_demo<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">ubuntu_installation<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">ubuntu_installation<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">English </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Install Ubuntu Server </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Choose the language </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Select your location </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Detect keyboard layout? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Country of origin for the keyboard </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Keyboard layout </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">No network interfaces detected </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Hostname: </span>
		<span className="blue ">for 30s with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">BACKSPACE </span>
		<span className="blue ">36 times </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"my-ubuntu" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Full name for the new user </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"my-ubuntu-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Username for your account </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Choose a password for the new user </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Re-enter password to verify </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Use weak password? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Encrypt your home directory? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Select your timezone </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Partitioning method </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Select disk to partition </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Write the changes to disks and configure LVM? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Amount of volume group to use for guided partitioning </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Write the changes to disks? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">LEFT </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">HTTP proxy information </span>
		<span className="blue ">for 3m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">How do you want to manage upgrades </span>
		<span className="blue ">for 6m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Choose software to install </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Install the GRUB boot loader to the master boot record? </span>
		<span className="blue ">for 10m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Installation complete </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Unplugging dvd </span>
		<span className="yellow "> </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">login: </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"my-ubuntu-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Password: </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Welcome to Ubuntu </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">ubuntu_installation</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="green bold">[ 33%] Test </span>
		<span className="yellow bold">ubuntu_installation</span>
		<span className="green bold"> PASSED in 0h:4m:24s<br/></span>
		<span className="blue ">[ 33%] Preparing the environment for test </span>
		<span className="yellow ">guest_additions_installation<br/></span>
		<span className="blue ">[ 33%] Running test </span>
		<span className="yellow ">guest_additions_installation<br/></span>
		<span className="blue ">[ 33%] Plugging dvd </span>
		<span className="yellow ">/opt/iso/testo-guest-additions.iso </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"sudo su" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">password for my-ubuntu-login </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">root@my-ubuntu </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"mount /dev/cdrom /media" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">mounting read-only </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"dpkg -i /media/*.deb" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Waiting </span>
		<span className="yellow ">Setting up testo-guest-additions </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Typing </span>
		<span className="yellow ">"umount /media" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Sleeping in virtual machine </span>
		<span className="yellow ">my_ubuntu</span>
		<span className="blue "> for 2s<br/></span>
		<span className="blue ">[ 33%] Unplugging dvd </span>
		<span className="yellow "> </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[ 33%] Taking snapshot </span>
		<span className="yellow ">guest_additions_installation</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="green bold">[ 67%] Test </span>
		<span className="yellow bold">guest_additions_installation</span>
		<span className="green bold"> PASSED in 0h:0m:14s<br/></span>
		<span className="blue ">[ 67%] Preparing the environment for test </span>
		<span className="yellow ">guest_additions_demo<br/></span>
		<span className="blue ">[ 67%] Running test </span>
		<span className="yellow ">guest_additions_demo<br/></span>
		<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
		<span className="yellow ">my_ubuntu</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ echo Hello world<br/></span>
		<span className=" ">Hello world<br/></span>
		<span className=" ">+ echo from bash<br/></span>
		<span className=" ">from bash<br/></span>
		<span className="blue ">[ 67%] Executing python3 command in virtual machine </span>
		<span className="yellow ">my_ubuntu</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">Hello from python3!<br/></span>
		<span className="blue ">[ 67%] Taking snapshot </span>
		<span className="yellow ">guest_additions_demo</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">guest_additions_demo</span>
		<span className="green bold"> PASSED in 0h:0m:3s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:4m:42s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)
