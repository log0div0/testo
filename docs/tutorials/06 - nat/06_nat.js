
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span class="">	- ubuntu_installation<br/></span>
		<span class="">	- guest_additions_installation<br/></span>
		<span class="">	- guest_additions_demo<br/></span>
		<span class="">Do you confirm running them? [y/N]: y<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">ubuntu_installation<br/></span>
		<span class="magenta ">guest_additions_installation<br/></span>
		<span class="magenta ">check_internet<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">ubuntu_installation<br/></span>
		<span class="blue ">[  0%] Creating virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
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
		<span class="red bold">/home/alex/testo/hello_world.testo:34:3: Error while performing action wait No network interfaces detected timeout 5m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
		<span class="red bold">[ 33%] Test </span>
		<span class="yellow bold">ubuntu_installation</span>
		<span class="red bold"> FAILED in 0h:5m:12s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">ubuntu_installation<br/></span>
		<span class="magenta ">guest_additions_installation<br/></span>
		<span class="magenta ">check_internet<br/></span>
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
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
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
		<span class="red bold">/home/alex/testo/hello_world.testo:43:3: Error while performing action wait Select your timezone timeout 2m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
		<span class="red bold">[ 33%] Test </span>
		<span class="yellow bold">ubuntu_installation</span>
		<span class="red bold"> FAILED in 0h:2m:57s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">ubuntu_installation<br/></span>
		<span class="magenta ">guest_additions_installation<br/></span>
		<span class="magenta ">check_internet<br/></span>
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
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
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
		<span class="yellow ">Is this time zone correct? </span>
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
		<span class="green bold"> PASSED in 0h:5m:16s<br/></span>
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
		<span class="yellow ">"dpkg -i /media/testo-guest-additions*" </span>
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
		<span class="green bold"> PASSED in 0h:0m:15s<br/></span>
		<span class="blue ">[ 67%] Preparing the environment for test </span>
		<span class="yellow ">check_internet<br/></span>
		<span class="blue ">[ 67%] Running test </span>
		<span class="yellow ">check_internet<br/></span>
		<span class="blue ">[ 67%] Executing bash command in virtual machine </span>
		<span class="yellow ">my_ubuntu</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ apt update<br/></span>
		<span class=" "><br/></span>
		<span class=" ">WARNING: apt does not have a stable CLI interface. Use with caution in scripts.<br/></span>
		<span class=" "><br/></span>
		<span class=" ">Hit:1 http://security.ubuntu.com/ubuntu xenial-security InRelease<br/></span>
		<span class=" ">Hit:2 http://us.archive.ubuntu.com/ubuntu xenial InRelease<br/></span>
		<span class=" ">Hit:3 http://us.archive.ubuntu.com/ubuntu xenial-updates InRelease<br/></span>
		<span class=" ">Hit:4 http://us.archive.ubuntu.com/ubuntu xenial-backports InRelease<br/></span>
		<span class=" ">Reading package lists...<br/></span>
		<span class=" ">Building dependency tree...<br/></span>
		<span class=" ">Reading state information...<br/></span>
		<span class=" ">150 packages can be upgraded. Run 'apt list --upgradable' to see them.<br/></span>
		<span class="blue ">[ 67%] Taking snapshot </span>
		<span class="yellow ">check_internet</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">check_internet</span>
		<span class="green bold"> PASSED in 0h:0m:7s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:5m:39s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 3<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)