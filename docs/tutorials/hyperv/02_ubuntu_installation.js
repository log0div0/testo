
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal>
		<span className="">C:\Users\Testo&gt; testo run ubuntu_installation.testo<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">my_first_test<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Install Ubuntu Server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="red bold">C:/Users/Testo/testo/ubuntu_installation.testo:15:3: Caught abort action on virtual machine my_ubuntu with message: Stop here<br/><br/>C:/Users/Testo/testo/ubuntu_installation.testo:2:1: note: the virtual machine my_ubuntu was declared here<br/><br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="red bold"> FAILED in 0h:0m:6s<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="350px">
		<span className="">C:\Users\Testo&gt; sudo testo run ubuntu_installation.testo --stop_on_fail<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">my_first_test<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"Install Ubuntu Server" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="red bold">C:/Users/Testo/testo/ubuntu_installation.testo:16:3: Caught abort action on virtual machine my_ubuntu with message: Stop here<br/><br/>C:/Users/Testo/testo/ubuntu_installation.testo:2:1: note: the virtual machine my_ubuntu was declared here<br/><br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="red bold"> FAILED in 0h:0m:3s<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="350px">
		<span className="">C:\Users\Testo&gt; sudo testo run ubuntu_installation.testo --stop_on_fail<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">my_first_test<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Waiting </span>
		<span className="yellow ">"ALALA" </span>
		<span className="blue ">for 10s with interval 1s in virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="red bold">C:/Users/Testo/testo/ubuntu_installation.testo:14:3: Error while performing action wait "ALALA" timeout 10s on virtual machine my_ubuntu<br/>      - Timeout<br/><br/>/C:/Users/Testo/testo/ubuntu_installation.testo:2:1: note: the virtual machine my_ubuntu was declared here<br/><br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="red bold"> FAILED in 0h:0m:11s<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)
