
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Ввод номера шлюза.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine marginRight='1px' left="317px" top="153px" font='Continent4' color='red'>Установка</ConsoleTextLine>
			<ConsoleTextLine marginRight='1px' left="180px" top="200px" font='Continent4' color='blue'>
				Введите идентификатор шлюза: <ConsoleText marginRight='1px' font='Continent4' color='gray'>__________</ConsoleText>
			</ConsoleTextLine>
			<ConsoleTextLine marginRight='1px' left="342px" top="249px" font='Continent4' color='gray'>ОК</ConsoleTextLine>
			<ConsoleTextLine marginRight='1px' left="10px" bottom="0px"  font='Continent4' color='gray'>
				{"<Tab>/<Alt-Tab> "}
				<ConsoleText marginRight='1px' font='Continent4' color='white'>между элементами</ConsoleText>
				{"  |   <Space> "}
				<ConsoleText marginRight='1px' font='Continent4' color='white'>выбор</ConsoleText>
				{"   |  <F12> "}
				<ConsoleText marginRight='1px' font='Continent4' color='white'>следующий экран</ConsoleText>
			</ConsoleTextLine>
		</Background>
	)
}
