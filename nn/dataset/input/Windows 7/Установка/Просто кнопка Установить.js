
import React from 'react'
import path from 'path'
import {TextLine, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Просто кнопка Установить.png')

	return (
		<Background src={bgPath}>
			<TextLine left='115px' top='75px' style={{fontSize: '9pt', color: 'white', fontWeight: 'bold'}}>Установка Windows</TextLine>
			<TextLine x='400px' top='205px' style={{fontSize: '28pt', color: 'white'}}>Windows 7</TextLine>
			<TextLine left='345px' top='263px' style={{fontSize: '12pt', color: 'white'}}>Установить</TextLine>
			<TextLine left='120px' top='435px' style={{fontSize: '10pt', color: 'white'}}>Что следует знать перед выполнением установки Windows</TextLine>
			<TextLine left='120px' top='465px' style={{fontSize: '10pt', color: 'white'}}>Восстановление системы</TextLine>
			<TextLine left='120px' top='495px' style={{fontSize: '8pt', color: 'white'}}>Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.</TextLine>
		</Background>
	)
}
