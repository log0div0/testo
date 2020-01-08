
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'black screen.png')

	return (
		<Background src={bgPath}>
			<Column left='0px' top='0px'>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ugen2.1: <Intel UHCI root HUB> at usbus2"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub2: <Intel UHCI root HUB, class 9/0, rev 1.00/1.00, addr 1> on usbus2"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ugen3.1: <Intel UHCI root HUB> at usbus3"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub3: <Intel UHCI root HUB, class 9/0, rev 2.00/1.00, addr 1> on usbus3"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub0: 2 ports with 2 removable, self powered"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub1: 2 ports with 2 removable, self powered"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub2: 2 ports with 2 removable, self powered"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ada0 at ata0 bus 0 scbus0 target 1 lun 0"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ada0: <QEMU HARDDISK 2.5+> ATA-7 device"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"cd0 at ata0 bus 0 scbus0 target 1 lun 0"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"cd0: <QEMU QEMU DVD-ROM 2.5+> Removable CD-ROM SCSI device"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"cd0: Serial Number QM00002"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"cd0: 16.700MB/s transfers (WDMA2, ATAPI 12bytes, PIO 65534bytes)"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"cd0: 2256MB (1155139 1048 byte selectors)"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ada0: Serial Number QM00001"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ada0: 16.700MB/s transfers (WDMA2, PIO 8192bytes)"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ada0: 5120MB (10485760 512 byte sectors)"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"ipcrypt: starting 1 task thread(s) *"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"Trying to mount root from cd9660:/dev/cd0 [ro]..."}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"gost: selftest"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"        ECB: ..........."}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"        CFB: ..............."}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"        IMIT: ........"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"        CTX: ...."}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"        0 errors"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"mountroot: unable to remount devfs under /dev (error 2)"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"mountroot: unable to unlink /dev/dev (error 2)"}</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>{"Нажмите Enter чтобы провести инсталляцию вручную"}</ConsoleTextLine>
				<ConsoleTextLine font='TerminusBold16' color='white'>{"uhub3: 6 ports with 6 removable, self powered"}</ConsoleTextLine>
			</Column>
		</Background>
	)
}
