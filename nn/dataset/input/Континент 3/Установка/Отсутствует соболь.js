
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'black screen.png')

	return (
		<Background src={bgPath}>
			<Column left='0px' top='0px'>
				<ConsoleTextLine font='Terminus16' color='gray'>All rights reserved.</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray' style={{marginBottom: '16px'}}>For info, please visit https://www.isc.org/software/dhcp/</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Listening on BPF/re0/52:54:00:ec:93:85</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Sending on   BPF/re0/52:54:00:ec:93:85</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Sending on   Socket/fallback</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>DHCPREQUEST on re0 to 255.255.255.255 port 67</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>DHCPACK from 192.168.122.1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>{"/sbin/dhcpclient-script: /bin/hostname: not found"}</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>New Hostname (re0):</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>{"/sbin/dhcpclient-script: /bin/hostname: not found"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>ifa_maintain_loopback_route: deletion failed for interface re0: 3</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>New IP Address (re0): 192.168.122.145</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>New Subnet Mask (re0): 255.255.255.0</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>New Broadcast Address (re0): 192.168.122.255</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>New Routers (re0): 192.168.122.1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>route: route has not been found</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>{"/sbin/dhcpclient-script: rm: not found"}</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>{"/sbin/dhcpclient-script: /sbin/resolvconf: not found"}</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>{"/sbin/dhcpclient-script: rm: not found"}</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>bound to 192.168.122.145 -- renewal in 1720 seconds.</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Не найден интерфейс с доступом к tftp-серверу</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Нажмите Enter чтобы провести инсталляцию вручную</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white' style={{marginBottom: '48px'}}>ifa_maintain_loopback_route: deletion failed for interface re0: 3</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Отсутствует электронный замок Соболь</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'>Дальнейшая работа будет производится без него</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='gray'> продолжить? (y/n):</ConsoleTextLine>
			</Column>
		</Background>
	)
}
