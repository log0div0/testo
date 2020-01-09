
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Таблица с сертификатами.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine left="0px" top="0px" font='Continent4' color='#0000a8'>2019-10-09 18:50:58 LOC |LAT|</ConsoleTextLine>
			<ConsoleTextLine left="500px" top="0px" font='Continent4' color='#0000a8'>Континент</ConsoleTextLine>
			<ConsoleTextLine left="480px" top="30px" font='Continent4' color='white'>Сертификаты УЦ</ConsoleTextLine>
			<Column left="8px" top="64px">
				<ConsoleTextLine font='Continent4' color='white'  > Кому выдан               | Кем выдан                                   | Тип     | Действителен с      | Действителен по    </ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='white'  >-----------------------------------------------------------------------------------------------------------------------------</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#0000a8'>Доверенный Издатель КБ    | Доверенный Издатель КБ                      | ca      | 2017-05-30 13:34:59 | 2028-05-30 13:44:24</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#adaaad'>Root                      | Root                                        | ca      | 2020-01-09 11:44:49 | 2025-01-07 11:44:49</ConsoleTextLine>
			</Column>
			<ConsoleTextLine left="10px" bottom="0px" font='Continent4' color='#0000a8'>
				Стрелки, PgUp, PgDn - навигация, F2 - Выпуск серт., F3 - Импорт, F5 - Импорт ключа, DEL - удалить, ESC - выход
			</ConsoleTextLine>
		</Background>
	)
}
