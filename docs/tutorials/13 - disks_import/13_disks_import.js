
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="400px">
		<span class="">user$ sudo testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR /var/lib/libvirt/images<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Creating virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Handmade </span>
		<span class="blue ">for 3m with interval 1s in virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">check_handmane_ubuntu</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">check_handmane_ubuntu</span>
		<span class="green bold"> PASSED in 0h:1m:13s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:1m:13s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="400px">
		<span class="">user$ sudo testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR /var/lib/libvirt/images<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">check_handmane_ubuntu<br/></span>
		<span class="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:0m:0s<br/></span>
		<span class="blue bold">UP-TO-DATE: 1<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="400px">
		<span class="">user$ sudo testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR /var/lib/libvirt/images<br/></span>
		<span class="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span class=""> - check_handmane_ubuntu<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Creating virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">check_handmane_ubuntu<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">Handmade </span>
		<span class="blue ">for 3m with interval 1s in virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="blue ">[  0%] Taking snapshot </span>
		<span class="yellow ">check_handmane_ubuntu</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">ubuntu_desktop<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">check_handmane_ubuntu</span>
		<span class="green bold"> PASSED in 0h:1m:15s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:1m:15s<br/></span>
		<span class="blue bold">UP-TO-DATE: 0<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)
