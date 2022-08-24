
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal>
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">my_first_test<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">my_first_test<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">my_first_test<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">English </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="red bold">/home/alex/testo/hello_world.testo:13:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
		<span class="red bold">[100%] Test </span>
		<span class="yellow bold">my_first_test</span>
		<span class="red bold"> FAILED in 0h:0m:4s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="350px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">my_first_test<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">my_first_test<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">my_first_test<br/></span>
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
		<span class="red bold">/home/alex/testo/hello_world.testo:13:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
		<span class="red bold">[100%] Test </span>
		<span class="yellow bold">my_first_test</span>
		<span class="red bold"> FAILED in 0h:0m:4s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="350px">
		<span class="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">my_first_test<br/></span>
		<span class="blue ">[  0%] Preparing the environment for test </span>
		<span class="yellow ">my_first_test<br/></span>
		<span class="blue ">[  0%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Running test </span>
		<span class="yellow ">my_first_test<br/></span>
		<span class="blue ">[  0%] Starting virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="blue ">[  0%] Waiting </span>
		<span class="yellow ">ALALA </span>
		<span class="blue ">for 10s with interval 1s in virtual machine </span>
		<span class="yellow ">my_ubuntu<br/></span>
		<span class="red bold">/home/alex/testo/hello_world.testo:13:3: Error while performing action wait ALALA timeout 10s on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
		<span class="red bold">[100%] Test </span>
		<span class="yellow bold">my_first_test</span>
		<span class="red bold"> FAILED in 0h:0m:11s<br/></span>
		<span class="">user$ </span>
	</Terminal>
)
