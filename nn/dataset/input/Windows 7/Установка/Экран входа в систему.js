
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Экран входа в систему.png')

	return (
		<Background src={bgPath}>
			<TextLine left='16px' top='10px' style={{fontSize: '10pt', color: 'white'}}>RU</TextLine>
			<TextLine left='370px' top='370px' style={{fontSize: '18pt', color: 'white', textShadow: '0px 3px 4px #676767'}}>Петя</TextLine>
			<TextLine left='294px' top='414px' style={{fontSize: '9pt', color: 'grey'}}>Пароль</TextLine>
			<TextLine left='295px' top='535px' style={{fontSize: '18pt', color: 'white', textShadow: '0px 3px 4px #676767'}}>Windows 7 <Text style={{fontSize: '14pt'}}>Максимальная</Text></TextLine>
		</Background>
	)
}
