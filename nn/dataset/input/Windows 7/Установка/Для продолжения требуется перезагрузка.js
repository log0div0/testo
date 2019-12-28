
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Для продолжения требуется перезагрузка.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '12pt', color: '#003399'}}>Для продолжения требуется перезагрузка Windows</TextLine>
			<TextLine left='128px' top='143px' style={{fontSize: '9pt'}}>Перезагрузка через 5 сек.</TextLine>
			<TextLine x='627px' y='472px' style={{fontSize: '9pt', color: '#0066cc'}}>Перезагрузить сейчас</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
