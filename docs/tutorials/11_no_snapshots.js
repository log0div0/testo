
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="620px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">test_ping<br/></span>
		<span className="magenta ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 80%] Preparing the environment for test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Running test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
		<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.037 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.043 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.044 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
		<span className=" "><br/></span>
		<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
		<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span className=" ">rtt min/avg/max/mdev = 0.037/0.045/0.057/0.008 ms<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
		<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.038 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.041 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.036 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.044 ms<br/></span>
		<span className=" "><br/></span>
		<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
		<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3997ms<br/></span>
		<span className=" ">rtt min/avg/max/mdev = 0.036/0.041/0.046/0.003 ms<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="green bold">[ 90%] Test </span>
		<span className="yellow bold">test_ping</span>
		<span className="green bold"> PASSED in 0h:0m:10s<br/></span>
		<span className="blue ">[ 90%] Preparing the environment for test </span>
		<span className="yellow ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Running test </span>
		<span className="yellow ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ echo 'Hello from client!'<br/></span>
		<span className="blue ">[ 90%] Calling macro </span>
		<span className="yellow ">process_flash(</span>
		<span className="yellow ">flash_name="exchange_flash"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">command="cp /tmp/copy_me_to_server.txt /media"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Plugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Sleeping in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> for 5s<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ mount /dev/sdb1 /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ umount /media<br/></span>
		<span className="blue ">[ 90%] Unplugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Calling macro </span>
		<span className="yellow ">process_flash(</span>
		<span className="yellow ">flash_name="exchange_flash"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">command="cp /media/copy_me_to_server.txt /tmp"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Plugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 5s<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ mount /dev/sdb1 /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ umount /media<br/></span>
		<span className="blue ">[ 90%] Unplugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
		<span className=" ">Hello from client!<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">exchange_files_with_flash</span>
		<span className="green bold"> PASSED in 0h:0m:18s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:28s<br/></span>
		<span className="blue bold">UP-TO-DATE: 8<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="350px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">test_ping<br/></span>
		<span className="magenta ">exchange_files_with_flash<br/></span>
		<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
		<span className="blue bold">UP-TO-DATE: 10<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="blue ">[ 67%] Preparing the environment for test </span>
		<span className="yellow ">client_unplug_nat<br/></span>
		<span className="blue ">[ 67%] Restoring snapshot </span>
		<span className="yellow ">client_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Running test </span>
		<span className="yellow ">client_unplug_nat<br/></span>
		<span className="blue ">[ 67%] Calling macro </span>
		<span className="yellow ">unplug_nic(</span>
		<span className="yellow ">hostname="client"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">login="client-login"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">nic_name="nat"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">password="1111"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Shutting down virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 1m<br/></span>
		<span className="blue ">[ 67%] Unplugging nic </span>
		<span className="yellow ">nat </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Starting virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"client login:" </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"client-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"Password:" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Waiting </span>
		<span className="yellow ">"Welcome to Ubuntu" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 67%] Taking snapshot </span>
		<span className="yellow ">client_unplug_nat</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">client_unplug_nat</span>
		<span className="green bold"> PASSED in 0h:0m:25s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:25s<br/></span>
		<span className="blue bold">UP-TO-DATE: 2<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="250px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
		<span className="blue bold">UP-TO-DATE: 3<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal5 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_prepare<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="blue ">[ 60%] Preparing the environment for test </span>
		<span className="yellow ">client_unplug_nat<br/></span>
		<span className="blue ">[ 60%] Restoring snapshot </span>
		<span className="yellow ">client_install_guest_additions</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Running test </span>
		<span className="yellow ">client_unplug_nat<br/></span>
		<span className="blue ">[ 60%] Calling macro </span>
		<span className="yellow ">unplug_nic(</span>
		<span className="yellow ">hostname="client"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">login="client-login"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">nic_name="nat"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">password="1111"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Shutting down virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 1m<br/></span>
		<span className="blue ">[ 60%] Unplugging nic </span>
		<span className="yellow ">nat </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Starting virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Waiting </span>
		<span className="yellow ">"client login:" </span>
		<span className="blue ">for 2m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Typing </span>
		<span className="yellow ">"client-login" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Waiting </span>
		<span className="yellow ">"Password:" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Typing </span>
		<span className="yellow ">"1111" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 60%] Waiting </span>
		<span className="yellow ">"Welcome to Ubuntu" </span>
		<span className="blue ">for 1m with interval 1s in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="green bold">[ 80%] Test </span>
		<span className="yellow bold">client_unplug_nat</span>
		<span className="green bold"> PASSED in 0h:0m:25s<br/></span>
		<span className="blue ">[ 80%] Preparing the environment for test </span>
		<span className="yellow ">client_prepare<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">initial</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="blue ">[ 80%] Running test </span>
		<span className="yellow ">client_prepare<br/></span>
		<span className="blue ">[ 80%] Calling macro </span>
		<span className="yellow ">process_flash(</span>
		<span className="yellow ">flash_name="exchange_flash"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">command="cp /media/rename_net.sh /opt/rename_net.sh"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Plugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Sleeping in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> for 5s<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ mount /dev/sdb1 /media<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cp /media/rename_net.sh /opt/rename_net.sh<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ umount /media<br/></span>
		<span className="blue ">[ 80%] Unplugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ chmod +x /opt/rename_net.sh<br/></span>
		<span className=" ">+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/></span>
		<span className=" ">Renaming success<br/></span>
		<span className=" ">+ ip a a 192.168.1.2/24 dev server_side<br/></span>
		<span className=" ">+ ip l s server_side up<br/></span>
		<span className=" ">+ ip ad<br/></span>
		<span className=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
		<span className=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span className=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
		<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span className=" ">    inet6 ::1/128 scope host <br/></span>
		<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span className=" ">2: server_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
		<span className=" ">    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/></span>
		<span className=" ">    inet 192.168.1.2/24 scope global server_side<br/></span>
		<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span className=" ">    inet6 fe80::5054:ff:fe00:aa/64 scope link tentative <br/></span>
		<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">client_prepare</span>
		<span className="green bold"> PASSED in 0h:0m:8s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:34s<br/></span>
		<span className="blue bold">UP-TO-DATE: 3<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal6 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">test_ping<br/></span>
		<span className="magenta ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 80%] Preparing the environment for test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Running test </span>
		<span className="yellow ">test_ping<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
		<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.037 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.043 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.044 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
		<span className=" "><br/></span>
		<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
		<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span className=" ">rtt min/avg/max/mdev = 0.037/0.045/0.057/0.008 ms<br/></span>
		<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
		<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.038 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.041 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.036 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
		<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.044 ms<br/></span>
		<span className=" "><br/></span>
		<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
		<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3997ms<br/></span>
		<span className=" ">rtt min/avg/max/mdev = 0.036/0.041/0.046/0.003 ms<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Taking snapshot </span>
		<span className="yellow ">test_ping</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="green bold">[ 90%] Test </span>
		<span className="yellow bold">test_ping</span>
		<span className="green bold"> PASSED in 0h:0m:10s<br/></span>
		<span className="blue ">[ 90%] Preparing the environment for test </span>
		<span className="yellow ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="blue ">[ 90%] Restoring snapshot </span>
		<span className="yellow ">client_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Running test </span>
		<span className="yellow ">exchange_files_with_flash<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ echo 'Hello from client!'<br/></span>
		<span className="blue ">[ 90%] Calling macro </span>
		<span className="yellow ">process_flash(</span>
		<span className="yellow ">flash_name="exchange_flash"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">command="cp /tmp/copy_me_to_server.txt /media"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Plugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Sleeping in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> for 5s<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ mount /dev/sdb1 /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">client</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ umount /media<br/></span>
		<span className="blue ">[ 90%] Unplugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Calling macro </span>
		<span className="yellow ">process_flash(</span>
		<span className="yellow ">flash_name="exchange_flash"</span>
		<span className="yellow ">, </span>
		<span className="yellow ">command="cp /media/copy_me_to_server.txt /tmp"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Plugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">into virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 5s<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ mount /dev/sdb1 /media<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ umount /media<br/></span>
		<span className="blue ">[ 90%] Unplugging flash drive </span>
		<span className="yellow ">exchange_flash </span>
		<span className="blue ">from virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> with timeout 10m<br/></span>
		<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
		<span className=" ">Hello from client!<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">client<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 90%] Taking snapshot </span>
		<span className="yellow ">exchange_files_with_flash</span>
		<span className="blue "> for flash drive </span>
		<span className="yellow ">exchange_flash<br/></span>
		<span className="green bold">[100%] Test </span>
		<span className="yellow bold">exchange_files_with_flash</span>
		<span className="green bold"> PASSED in 0h:0m:18s<br/></span>
		<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:28s<br/></span>
		<span className="blue bold">UP-TO-DATE: 8<br/></span>
		<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span className="red bold">FAILED: 0<br/></span>
		<span className="">user$ </span>
	</Terminal>
)

export const terminal7 = (
	<Terminal height="350px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec test_ping --assume_yes<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">client_install_ubuntu<br/></span>
		<span className="magenta ">client_install_guest_additions<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">test_ping<br/></span>
		<span className="magenta ">client_unplug_nat<br/></span>
		<span className="magenta ">client_prepare<br/></span>
		<span className="magenta ">server_unplug_nat<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">exchange_files_with_flash<br/></span>
		...<br/>
		<span className="">user$ </span>
	</Terminal>
)
