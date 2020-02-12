
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Пожалуйста, извлеките CD-ROM.png')

	return (
		<Background src={bgPath}>
			<Column left='0px' top='0px'>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='red'>FAILED</ConsoleText>] Failed unmounting Mount unit for core, revision 6350.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Network Time Synchronization.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='red'>FAILED</ConsoleText>] Failed unmounting Mount unit for subiquity, revision 664.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Load/Save Random Seed.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Update UTMP about System Boot/Shutdown.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Create Volatile Files and Directories.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped target Local File Systems.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /target/run/cdrom...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /rofs...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /lib/modules...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /tmp...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='red'>FAILED</ConsoleText>] Failed unmounting /lib/modules.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Unmounted /rofs.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Unmounted /tmp.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Unmounted /target/run/cdrom.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /target/run...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Unmounted /target/run.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Unmounting /target...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped target Swap.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Unmounted /target.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Reached target Unmount All Filesystems.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped target Local File Systems (Pre).</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Remount Root and Kernel File Systems..</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Create Static Device Nodes in /dev.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Reached target Shutdown.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Starting Shuts down the "live" preinstalled system cleanly...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Stopping Monitoring of LVM2 mirrors, snapshots etc. using dmeventd or progress polling...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped Monitoring of LVM2 mirrors, snapshots etc. using dmeventd or progress polling.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>         Stopping LVM2 metadata daemon...</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>[<ConsoleText font='Fixed16' color='green'>  OK  </ConsoleText>] Stopped LVM2 metadata daemon.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>Please remove the installation medium, then press ENTER:</ConsoleTextLine>
			</Column>
		</Background>
	)
}
