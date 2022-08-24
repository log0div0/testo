
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_install_ubuntu --assume_yes<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Calling macro </span>
		<span class="yellow ">install_ubuntu(</span>
		<span class="yellow ">hostname="server"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">login="server-login"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">password="1111"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">is_weak_password=""</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">server<br/></span>
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
		<span class="yellow ">Primary network interface </span>
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
		<span class="yellow ">Encrypt your home directory? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="red bold">/home/alex/testo/macros.testo:25:2: Error while performing action wait Encrypt your home directory? on virtual machine server:<br/>	-Timeout<br/></span>
		<span class="red bold">[100%] Test </span>
		<span class="yellow bold">server_install_ubuntu</span>
		<span class="red bold"> FAILED in 0h:2m:3s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_install_ubuntu --assume_yes<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">server_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Calling macro </span>
		<span class="yellow ">install_ubuntu(</span>
		<span class="yellow ">hostname="server"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">login="server-login"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">password="ThisIsStrongPassword"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">is_weak_password=""</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">server<br/></span>
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
		<span class="yellow ">Primary network interface </span>
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
		<span class="yellow ">"ThisIsStrongPassword" </span>
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
		<span class="yellow ">"ThisIsStrongPassword" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
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
		<span class="yellow ">"ThisIsStrongPassword" </span>
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
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">server_install_ubuntu</span>
		<span class="green bold"> PASSED in 0h:5m:34s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:5m:34s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec client_install_ubuntu --assume_yes<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">client_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">client_install_ubuntu<br/></span>
		<span class="blue ">[  0%] Calling macro </span>
		<span class="yellow ">install_ubuntu(</span>
		<span class="yellow ">hostname="client"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">login="client-login"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">password="1111"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">is_weak_password="yes"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">English </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install Ubuntu Server </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose the language </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select your location </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Detect keyboard layout? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Country of origin for the keyboard </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Keyboard layout </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Primary network interface </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Hostname: </span>
		<span class="blue ">for 5m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">BACKSPACE </span>
		<span class="blue ">36 times </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"client" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Full name for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"client-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Username for your account </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose a password for the new user </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Re-enter password to verify </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Use weak password? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Encrypt your home directory? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Is this time zone correct? </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Partitioning method </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Select disk to partition </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks and configure LVM? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Amount of volume group to use for guided partitioning </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Write the changes to disks? </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">LEFT </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">HTTP proxy information </span>
		<span class="blue ">for 3m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">How do you want to manage upgrades </span>
		<span class="blue ">for 6m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Choose software to install </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Install the GRUB boot loader to the master boot record? </span>
		<span class="blue ">for 10m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Installation complete </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Unplugging dvd </span>
		<span class="yellow "> </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">client login: </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"client-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Password: </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">on virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Welcome to Ubuntu </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">client_install_ubuntu</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">client_install_ubuntu</span>
		<span class="green bold"> PASSED in 0h:5m:16s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:5m:16s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="300px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec \*_install_ubuntu --assume_yes<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="">...<br/></span>
		<span class="blue bold">PROCESSED TOTAL 2 TESTS IN 0h:11m:14s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)