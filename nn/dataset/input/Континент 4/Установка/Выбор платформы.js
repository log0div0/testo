
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выбор платформы.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine marginRight='1px' left="260px" top="88px" font='Continent4' color='#ff5757'>Выберите тип платформы</ConsoleTextLine>
			<Column left='252px' top='120px'>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='grey'>
					<ConsoleText marginRight='1px' font='Continent4' color='white'>Настраиваемая</ConsoleText>/Custom
				</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-10 (LN010A)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-25 (MSS1151)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-50 (LN010C)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-50M (LN010M)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-100 (MSS102A)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-400 (MSS021)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-500 (LN015B)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-500F (LN015C)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-500M (LN015M)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-600 (DV030A)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-600M (DV030M)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-800F (DV030B)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-1000 (MSS021)</ConsoleTextLine>
				<ConsoleTextLine marginRight='1px' font='Continent4' color='black'>IPC-1000F (MSS021)</ConsoleTextLine>
			</Column>
			<ConsoleTextLine marginRight='1px' left="10px" bottom="0px"  font='Continent4' color='grey'>
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
