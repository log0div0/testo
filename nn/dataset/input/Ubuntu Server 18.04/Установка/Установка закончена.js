
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Установка закончена.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine font='Fixed16' color='white' left='105px' top='15px'>Installation complete!</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='443px' top='62px'>Finished install!</ConsoleTextLine>
			<Column left='113px' top='80px'>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring partition: part-1</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring format: fs-0</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring mount: mount-0</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  configuring network</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'curtin net-meta auto'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>      curtin command net-meta</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  writing install sources to disk</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'curtin extract'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>      curtin command extract</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        acquiring and extracting image from cp:///media/filesystem</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  configuring installed system</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'mount -t tmpfs tmpfs /target/run/cdrom'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'mkdir -p /target/run/cdrom'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'mount --bind /cdrom /target/run/cdrom'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'curtin curthooks'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>      curtin command cirthooks</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring apt configuring apt</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        installing missing packages</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring iscsci service</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring raid (mdadm) service</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        installing kernel</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        setting up swap</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        apply networking config</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        writing etc/fstab</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring multipath</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        updating packages on target system</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>        configuring pollinate user-agent on target</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  finalizing installation</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>    running 'curtin hook'</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>      curtin command hook</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  executing late commands</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>final system configuration</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  configuring cloud-init</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  installing OpenSSH server</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='white'>  cleaning up apt configuration</ConsoleTextLine>
			</Column>
			<ConsoleTextLine font='Fixed16' color='white' left='440px' top='671px'>[ View full log ]</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='440px' top='687px'>[ Reboot Now    ]</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='480px' top='719px'>12 / 12</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='105px' bottom='0px'>Thank you for using Ubuntu!</ConsoleTextLine>
		</Background>
	)
}
