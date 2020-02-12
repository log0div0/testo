
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Windows завершает применение параметров.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='270px' y='207px' style={{fontSize: '18pt', letterSpacing: '-1px'}}><Text style={{letterSpacing: '-2px', fontSize: '26pt'}}>Windows 7</Text> Максимальная</TextLine>
			<TextLine x='403px' y='325px' style={{fontSize: '9pt'}}>Windows завершает применение параметров</TextLine>
		</Background>
	)
}
