
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span className="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span className="">	- ubuntu_installation<br/></span>
		<span className="">	- guest_additions_installation<br/></span>
		<span className="">	- guest_additions_demo<br/></span>
		<span className="">Do you confirm running them? [y/N]: y<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">ubuntu_installation<br/></span>
		<span className="magenta ">guest_additions_installation<br/></span>
		<span className="magenta ">check_internet<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">ubuntu_installation<br/></span>
		<span className="blue ">[  0%] Creating virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
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
		<span className="red bold">/home/alex/testo/hello_world.testo:34:3: Error while performing action wait No network interfaces detected timeout 5m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
		<span className="red bold">[ 33%] Test </span>
		<span className="yellow bold">ubuntu_installation</span>
		<span className="red bold"> FAILED in 0h:5m:12s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">ubuntu_installation<br/></span>
		<span className="magenta ">guest_additions_installation<br/></span>
		<span className="magenta ">check_internet<br/></span>
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
		<span className="yellow ">Hostname: </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
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
		<span className="red bold">/home/alex/testo/hello_world.testo:43:3: Error while performing action wait Select your timezone timeout 2m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
		<span className="red bold">[ 33%] Test </span>
		<span className="yellow bold">ubuntu_installation</span>
		<span className="red bold"> FAILED in 0h:2m:57s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">ubuntu_installation<br/></span>
		<span className="magenta ">guest_additions_installation<br/></span>
		<span className="magenta ">check_internet<br/></span>
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
		<span className="yellow ">Hostname: </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
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
		<span className="yellow ">Is this time zone correct? </span>
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
		<span className="green bold"> PASSED in 0h:5m:16s<br/></span>
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
		<span className="yellow ">"dpkg -i /media/testo-guest-additions*" </span>
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
		<span className="green bold"> PASSED in 0h:0m:15s<br/></span>
		<span className="blue ">[ 67%] Preparing the environment for test </span>
		<span className="yellow ">check_internet<br/></span>
		<span className="blue ">[ 67%] Running test </span>
		<span className="yellow ">check_internet<br/></span>
		<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
		<span className="yellow ">my_ubuntu</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ apt update<br/></span>
		<span className=" "><br/></span>
		<span className=" ">WARNING: apt does not have a stable CLI interface. Use with caution in scripts.<br/></span>
		<span className=" "><br/></span>
		<span className=" ">Hit:1 http://security.ubuntu.com/ubuntu xenial-security InRelease<br/></span>
		<span className=" ">Hit:2 http://us.archive.ubuntu.com/ubuntu xenial InRelease<br/></span>
		<span className=" ">Hit:3 http://us.archive.ubuntu.com/ubuntu xenial-updates InRelease<br/></span>
		<span className=" ">Hit:4 http://us.archive.ubuntu.com/ubuntu xenial-backports InRelease<br/></span>
		<span className=" ">Reading package lists...<br/></span>
		<span className=" ">Building dependency tree...<br/></span>
		<span className=" ">Reading state information...<br/></span>
		<span className=" ">150 packages can be upgraded. Run 'apt list --upgradable' to see them.<br/></span>
		<span className="blue ">[ 67%] Taking snapshot </span>
		<span className="yellow ">check_internet</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">check_internet</span>
		<span className="green bold"> PASSED in 0h:0m:7s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:5m:39s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)
