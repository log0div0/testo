
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="400px">
		<span className="">user$ sudo testo run mouse.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">English </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Try Ubuntu without installing </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Welcome </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="red bold">/home/alex/testo/mouse.testo:28:3: Caught abort action on virtual machine ubuntu_desktop with message: stop here<br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">install_ubuntu</span>
		<span className="red bold"> FAILED in 0h:0m:46s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="620px">
		<span className="">user$ sudo testo run mouse.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">install_ubuntu<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">English </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Try Ubuntu without installing </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Welcome </span>
		<span className="blue ">for 5m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Continue.center_bottom() </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Keyboard layout </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Continue.center_bottom() </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Updates and other software </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Minimal installation </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Download updates while installing </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Continue </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Installation type </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse moving </span>
		<span className="blue ">on coordinates </span>
		<span className="yellow ">0 0 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Install Now.center_bottom() </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">Write the changes to disks? </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="blue ">[  0%] Mouse clicking </span>
		<span className="blue ">on </span>
		<span className="yellow ">Continue.center_bottom() </span>
		<span className="blue ">with timeout 1m in virtual machine </span>
		<span className="yellow ">ubuntu_desktop<br/></span>
		<span className="red bold">/home/alex/testo/mouse.testo:42:45: Error while performing action click Continue.center_bottom() on virtual machine ubuntu_desktop:<br/>	-Can't apply specifier "center_bottom": there's more than one object<br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">install_ubuntu</span>
		<span className="red bold"> FAILED in 0h:1m:11s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)