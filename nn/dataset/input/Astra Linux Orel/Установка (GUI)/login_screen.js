
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Row, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'login_screen.png')

	return (
		<Background src={bgPath}>
			<TextLine left='190px' y='20px' style={{color: '#bbbbbb', fontSize: '11pt'}}>Вход в astra</TextLine>
			<TextLine right='190px' y='20px' style={{color: '#bbbbbb', fontSize: '11pt'}}>среда, 18 декабря 2019 г. 14:59:36 MSK</TextLine>
			<TextLine left='68px' y='382px' style={{color: 'black', fontSize: '12pt'}}>user</TextLine>
			<TextLine left='318px' y='370px' style={{color: '#bbbbbb', fontSize: '12pt'}}>Имя:</TextLine>
			<TextLine left='318px' y='395px' style={{color: '#bbbbbb', fontSize: '12pt'}}>Пароль:</TextLine>
			<TextLine right='80px' y='350px' style={{color: '#bbbbbb', fontSize: '12pt'}}>Тип сессии</TextLine>
			<TextLine right='80px' y='415px' style={{color: '#bbbbbb', fontSize: '12pt'}}>Меню</TextLine>
			<TextLine x='972px' y='729px' style={{color: '#bbbbbb', fontSize: '12pt'}}>En</TextLine>
		</Background>
	)
}
