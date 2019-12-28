
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Row, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Start.png')

	return (
		<Background src={bgPath}>
			<TextLine x='113px' y='45px' style={{fontSize: '19pt'}}>ASTRALINUX</TextLine>
			<TextLine x='113px' y='65px' style={{fontSize: '11pt'}}>common edition</TextLine>
			<Column right='25px' top='15px'>
				<TextLine style={{fontSize: '11pt'}}>Операционная система</TextLine>
				<TextLine style={{fontSize: '11pt'}}>общего назначения</TextLine>
				<TextLine style={{fontSize: '11pt', fontWeight: 'bold'}}>Релиз "Орёл"</TextLine>
			</Column>
			<Column x='320px' top='222px'>
				<ConsoleTextLine font='Fixed16' color='white' style={{marginBottom: '4px'}}>Графическая установка</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='red' style={{marginBottom: '4px'}}>Установка</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='red' style={{marginBottom: '4px'}}>Режим восстановления</ConsoleTextLine>
			</Column>
			<TextLine left='25px' top='350px' style={{fontSize: '11pt', color: 'grey'}}>"Орёл - город воинской славы"</TextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='31px' top='405px'>Русский</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='black' left='31px' top='430px'>English</ConsoleTextLine>
			<Row left='15px' top='450px'>
				<ConsoleTextLine font='Fixed16' color='red' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F1</ConsoleText> Язык</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='red' style={{marginRight: '10px'}}><ConsoleText font='Fixed16' color='white'>F2</ConsoleText> Параметры</ConsoleTextLine>
			</Row>
		</Background>
	)
}
