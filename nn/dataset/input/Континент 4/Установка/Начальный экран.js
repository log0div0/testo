
import React from 'react'
import path from 'path'
import {TextLine, Text, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Начальный экран.png')

	return (
		<Background src={bgPath}>
			<TextLine left='260px' y='203px' style={{fontSize: '14pt', color: 'white'}}>
				<Text style={{fontWeight: 'bold'}}>КОД</Text> БЕЗОПАСНОСТИ
			</TextLine>
			<Column left='200px' top='280px'>
				<ConsoleTextLine font='Continent4' color='white'>Установить Континент 4.1.0.919</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#d4e9d2'>Тест памяти</ConsoleTextLine>
			</Column>
		</Background>
	)
}
