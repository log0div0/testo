
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="200px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo<br/></span>
		<span className="bold blue">PROCESSED TOTAL 0 TESTS IN 0h:0m:0s<br/></span>
		<span className="bold blue">UP-TO-DATE: 0<br/></span>
		<span className="bold green">RUN SUCCESSFULLY: 0<br/></span>
		<span className="bold red">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="400px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">my_first_test<br/></span>
		<span className="blue ">[  0%] Preparing the environment for test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Creating virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Running test </span>
		<span className="yellow ">my_first_test<br/></span>
		<span className="blue ">[  0%] Starting virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="blue ">[  0%] Taking snapshot </span>
		<span className="yellow ">my_first_test</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">my_ubuntu<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="green bold"> PASSED in 0h:0m:1s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:0m:1s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="500px">
		<span className="">user$ sudo testo run ~/testo/hello_world.testo<br/></span>
		<span className="">Because of the cache loss, Testo is scheduled to run the following tests:<br/></span>
		<span className="">	- my_first_test<br/></span>
		<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
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
		<span className="red bold">/home/alex/testo/hello_world.testo:14:3: Caught abort action on virtual machine my_ubuntu with message: Stop here<br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="red bold"> FAILED in 0h:0m:1s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 1 TESTS IN 0h:0m:1s<br/></span>
		<span className="blue bold">UP-TO-DATE: 0<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span className="red bold">FAILED: 1<br/></span>
		<span className="red ">	 -my_first_test<br/></span>
		<span>At least one of the tests failed<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal>
		<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
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
		<span className="red bold">/home/alex/testo/hello_world.testo:14:3: Caught abort action on virtual machine my_ubuntu with message: Stop here<br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">my_first_test</span>
		<span className="red bold"> FAILED in 0h:0m:1s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal5 = (
	<Terminal height="150px">
		<span className="">user$ sudo testo clean<br/></span>
		<span className="">Testo is about to erase the following entities:<br/></span>
		<span className="">Virtual machines:<br/></span>
		<span className="">		- my_ubuntu<br/></span>
		<span className=""><br/></span>
		<span className="">Do you confirm erasing these entities? [y/N]: y<br/></span>
		<span className="">Deleted virtual machine my_ubuntu<br/></span>
		<span className="">user$ </span>
	</Terminal>
)
