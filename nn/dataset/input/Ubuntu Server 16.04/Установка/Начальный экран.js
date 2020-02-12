
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Начальный экран.png')

	return (
		<Background src={bgPath}>
			<TextLine left='270px' top='120px' style={{fontSize: '22pt', color: 'white', textShadow: '0 0 3px white'}}>ubuntu</TextLine>
			<Column x='320px' top='220px'>
				<ConsoleTextLine font='Fixed16' color='white' style={{marginBottom: '4px'}}>Install Ubuntu Server</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Install Ubuntu Server with the HWE kernel</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Install MAAS Region Controller</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Install MAAS Rack Controller</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Check disc for defects</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Test Memory</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Boot from first hard disk</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginBottom: '4px'}}>Rescure a broken system</ConsoleTextLine>
			</Column>
			<Row left='15px' top='450px'>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F1</ConsoleText> Help</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F2</ConsoleText> Language</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F3</ConsoleText> Keymap</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F4</ConsoleText> Modes</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F5</ConsoleText> Accessibility</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='gray' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F6</ConsoleText> Other Options</ConsoleTextLine>
			</Row>
		</Background>
	)
}
