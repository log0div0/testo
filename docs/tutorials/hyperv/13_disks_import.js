
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="400px">
		<span className="">C:\Users\Testo&gt; testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR "C:\Users\Public\Documents\Hyper-V\Virtual hard disks"<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Creating virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Handmade" </span>
		<span className="blue ">for 3m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">check_handmane_ubuntu</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">check_handmane_ubuntu</span>
		<span className="green bold"> PASSED in 0h:0m:21s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:0m:21s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="400px">
		<span className="">C:\Users\Testo&gt; testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR "C:\Users\Public\Documents\Hyper-V\Virtual hard disks"<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">check_handmane_ubuntu<br/></span>
		<span className="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:0m:0s<br/></span>
		<span className="blue bold">UP-TO-DATE: 1<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="400px">
		<span className="">C:\Users\Testo&gt; testo run handmade.testo --stop_on_fail --param VM_DISK_POOL_DIR "C:\Users\Public\Documents\Hyper-V\Virtual hard disks"<br/></span>
		<span className="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span className=""> - check_handmane_ubuntu<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Creating virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">check_handmane_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Handmade </span>
		<span className="blue ">for 3m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">check_handmane_ubuntu</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">check_handmane_ubuntu</span>
		<span className="green bold"> PASSED in 0h:1m:15s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:1m:15s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)
