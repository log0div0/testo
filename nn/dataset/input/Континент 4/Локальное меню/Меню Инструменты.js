
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column, Row} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Меню Инструменты.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine left="0px" top="0px" font='Continent4' color='#0000a8'>2019-10-09 18:50:58 LOC |LAT|</ConsoleTextLine>
			<ConsoleTextLine left="500px" top="0px" font='Continent4' color='#0000a8'>Континент</ConsoleTextLine>
			<ConsoleTextLine left="450px" top="255px" font='Continent4' color='white'>Инструменты</ConsoleTextLine>
			<Column left='216px' top='288px'>
				<ConsoleTextLine font='Continent4' color='#0000a8'>Диагностика</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Резервное копирование и восстановление</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Отправка локальных изменений на ЦУС</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Создать узел безопасности</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Создать узел безопасности с резервным ЦУС</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Подтверждение изменений настроек УБ</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Экспорт конфигурации УБ на носитель</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Изменение пароля встроенного администратора</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Загрузка конфигурации с носителя</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Повторная инициализация</ConsoleTextLine>
				<ConsoleTextLine font='Continent4' color='#a8a8a8'>Возврат в предыдущее меню</ConsoleTextLine>
			</Column>
			<ConsoleTextLine left="10px" bottom="0px"  font='Continent4' color='#0000a8'>Стрелки, PgUp, PgDn - навигация, ESC - выход</ConsoleTextLine>
		</Background>
	)
}
