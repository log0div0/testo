
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="620px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">test_ping<br/></span>
		<span class="magenta ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 80%] Preparing the environment for test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Running test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.2 -c5<br/></span>
		<span class=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.037 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.043 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.044 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.2 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.037/0.045/0.057/0.008 ms<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.1 -c5<br/></span>
		<span class=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.038 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.041 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.036 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.044 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.1 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3997ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.036/0.041/0.046/0.003 ms<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="green bold">[ 90%] Test </span>
		<span class="yellow bold">test_ping</span>
		<span class="green bold"> PASSED in 0h:0m:10s<br/></span>
		<span class="blue ">[ 90%] Preparing the environment for test </span>
		<span class="yellow ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Running test </span>
		<span class="yellow ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ echo 'Hello from client!'<br/></span>
		<span class="blue ">[ 90%] Calling macro </span>
		<span class="yellow ">process_flash(</span>
		<span class="yellow ">flash_name="exchange_flash"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">command="cp /tmp/copy_me_to_server.txt /media"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Plugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Sleeping in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> for 5s<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ mount /dev/sdb1 /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ umount /media<br/></span>
		<span class="blue ">[ 90%] Unplugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Calling macro </span>
		<span class="yellow ">process_flash(</span>
		<span class="yellow ">flash_name="exchange_flash"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">command="cp /media/copy_me_to_server.txt /tmp"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Plugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Sleeping in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> for 5s<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ mount /dev/sdb1 /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ umount /media<br/></span>
		<span class="blue ">[ 90%] Unplugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
		<span class=" ">Hello from client!<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">exchange_files_with_flash</span>
		<span class="green bold"> PASSED in 0h:0m:18s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:28s<br/></span>
		<span class="blue bold">UP-TO-DATE: 8<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="350px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="magenta ">test_ping<br/></span>
		<span class="magenta ">exchange_files_with_flash<br/></span>
		<span class="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
		<span class="blue bold">UP-TO-DATE: 10<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal3 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Preparing the environment for test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Restoring snapshot </span>
		<span class="yellow ">client_install_guest_additions</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Running test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 67%] Calling macro </span>
		<span class="yellow ">unplug_nic(</span>
		<span class="yellow ">hostname="client"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">login="client-login"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">nic_name="nat"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">password="1111"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Shutting down virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 1m<br/></span>
		<span class="blue ">[ 67%] Unplugging nic </span>
		<span class="yellow ">nat </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Starting virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">"client login:" </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"client-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">"Password:" </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Waiting </span>
		<span class="yellow ">"Welcome to Ubuntu" </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 67%] Taking snapshot </span>
		<span class="yellow ">client_unplug_nat</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">client_unplug_nat</span>
		<span class="green bold"> PASSED in 0h:0m:25s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:25s<br/></span>
		<span class="blue bold">UP-TO-DATE: 2<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal4 = (
	<Terminal height="250px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
		<span class="blue bold">UP-TO-DATE: 3<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 0<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal5 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_prepare<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="blue ">[ 60%] Preparing the environment for test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 60%] Restoring snapshot </span>
		<span class="yellow ">client_install_guest_additions</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Running test </span>
		<span class="yellow ">client_unplug_nat<br/></span>
		<span class="blue ">[ 60%] Calling macro </span>
		<span class="yellow ">unplug_nic(</span>
		<span class="yellow ">hostname="client"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">login="client-login"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">nic_name="nat"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">password="1111"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Shutting down virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 1m<br/></span>
		<span class="blue ">[ 60%] Unplugging nic </span>
		<span class="yellow ">nat </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Starting virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Waiting </span>
		<span class="yellow ">"client login:" </span>
		<span class="blue ">for 2m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Typing </span>
		<span class="yellow ">"client-login" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Waiting </span>
		<span class="yellow ">"Password:" </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Typing </span>
		<span class="yellow ">"1111" </span>
		<span class="blue ">with interval 30ms in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Pressing key </span>
		<span class="yellow ">ENTER </span>
		<span class="blue ">in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 60%] Waiting </span>
		<span class="yellow ">"Welcome to Ubuntu" </span>
		<span class="blue ">for 1m with interval 1s in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="green bold">[ 80%] Test </span>
		<span class="yellow bold">client_unplug_nat</span>
		<span class="green bold"> PASSED in 0h:0m:25s<br/></span>
		<span class="blue ">[ 80%] Preparing the environment for test </span>
		<span class="yellow ">client_prepare<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">initial</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="blue ">[ 80%] Running test </span>
		<span class="yellow ">client_prepare<br/></span>
		<span class="blue ">[ 80%] Calling macro </span>
		<span class="yellow ">process_flash(</span>
		<span class="yellow ">flash_name="exchange_flash"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">command="cp /media/rename_net.sh /opt/rename_net.sh"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Plugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Sleeping in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> for 5s<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ mount /dev/sdb1 /media<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cp /media/rename_net.sh /opt/rename_net.sh<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ umount /media<br/></span>
		<span class="blue ">[ 80%] Unplugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ chmod +x /opt/rename_net.sh<br/></span>
		<span class=" ">+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/></span>
		<span class=" ">Renaming success<br/></span>
		<span class=" ">+ ip a a 192.168.1.2/24 dev server_side<br/></span>
		<span class=" ">+ ip l s server_side up<br/></span>
		<span class=" ">+ ip ad<br/></span>
		<span class=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
		<span class=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
		<span class=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 ::1/128 scope host <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">2: server_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
		<span class=" ">    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/></span>
		<span class=" ">    inet 192.168.1.2/24 scope global server_side<br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class=" ">    inet6 fe80::5054:ff:fe00:aa/64 scope link tentative <br/></span>
		<span class=" ">       valid_lft forever preferred_lft forever<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">client_prepare</span>
		<span class="green bold"> PASSED in 0h:0m:8s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:34s<br/></span>
		<span class="blue bold">UP-TO-DATE: 3<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 1<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal6 = (
	<Terminal height="600px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">test_ping<br/></span>
		<span class="magenta ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 80%] Preparing the environment for test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="blue ">[ 80%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Running test </span>
		<span class="yellow ">test_ping<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.2 -c5<br/></span>
		<span class=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.037 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.043 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.044 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.2 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.037/0.045/0.057/0.008 ms<br/></span>
		<span class="blue ">[ 80%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ ping 192.168.1.1 -c5<br/></span>
		<span class=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.038 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.041 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.036 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
		<span class=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.044 ms<br/></span>
		<span class=" "><br/></span>
		<span class=" ">--- 192.168.1.1 ping statistics ---<br/></span>
		<span class=" ">5 packets transmitted, 5 received, 0% packet loss, time 3997ms<br/></span>
		<span class=" ">rtt min/avg/max/mdev = 0.036/0.041/0.046/0.003 ms<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 80%] Taking snapshot </span>
		<span class="yellow ">test_ping</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="green bold">[ 90%] Test </span>
		<span class="yellow bold">test_ping</span>
		<span class="green bold"> PASSED in 0h:0m:10s<br/></span>
		<span class="blue ">[ 90%] Preparing the environment for test </span>
		<span class="yellow ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">server_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="blue ">[ 90%] Restoring snapshot </span>
		<span class="yellow ">client_prepare</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Running test </span>
		<span class="yellow ">exchange_files_with_flash<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ echo 'Hello from client!'<br/></span>
		<span class="blue ">[ 90%] Calling macro </span>
		<span class="yellow ">process_flash(</span>
		<span class="yellow ">flash_name="exchange_flash"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">command="cp /tmp/copy_me_to_server.txt /media"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Plugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Sleeping in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> for 5s<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ mount /dev/sdb1 /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">client</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ umount /media<br/></span>
		<span class="blue ">[ 90%] Unplugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Calling macro </span>
		<span class="yellow ">process_flash(</span>
		<span class="yellow ">flash_name="exchange_flash"</span>
		<span class="yellow ">, </span>
		<span class="yellow ">command="cp /media/copy_me_to_server.txt /tmp"</span>
		<span class="yellow ">)</span>
		<span class="blue "> in virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Plugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">into virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Sleeping in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> for 5s<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ mount /dev/sdb1 /media<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ umount /media<br/></span>
		<span class="blue ">[ 90%] Unplugging flash drive </span>
		<span class="yellow ">exchange_flash </span>
		<span class="blue ">from virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Executing bash command in virtual machine </span>
		<span class="yellow ">server</span>
		<span class="blue "> with timeout 10m<br/></span>
		<span class=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
		<span class=" ">Hello from client!<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">client<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for virtual machine </span>
		<span class="yellow ">server<br/></span>
		<span class="blue ">[ 90%] Taking snapshot </span>
		<span class="yellow ">exchange_files_with_flash</span>
		<span class="blue "> for flash drive </span>
		<span class="yellow ">exchange_flash<br/></span>
		<span class="green bold">[100%] Test </span>
		<span class="yellow bold">exchange_files_with_flash</span>
		<span class="green bold"> PASSED in 0h:0m:18s<br/></span>
		<span class="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:28s<br/></span>
		<span class="blue bold">UP-TO-DATE: 8<br/></span>
		<span class="green bold">RUN SUCCESSFULLY: 2<br/></span>
		<span class="red bold">FAILED: 0<br/></span>
		<span class="">user$ </span>
	</Terminal>
)

export const terminal7 = (
	<Terminal height="350px">
		<span class="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec test_ping --assume_yes<br/></span>
		<span class="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span class="magenta ">server_install_ubuntu<br/></span>
		<span class="magenta ">server_install_guest_additions<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">client_install_ubuntu<br/></span>
		<span class="magenta ">client_install_guest_additions<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="blue bold">TESTS TO RUN:<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="magenta ">test_ping<br/></span>
		<span class="magenta ">client_unplug_nat<br/></span>
		<span class="magenta ">client_prepare<br/></span>
		<span class="magenta ">server_unplug_nat<br/></span>
		<span class="magenta ">server_prepare<br/></span>
		<span class="magenta ">exchange_files_with_flash<br/></span>
		...<br/>
		<span class="">user$ </span>
	</Terminal>
)
