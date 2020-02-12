
import React from 'react'
import path from 'path'
import {TextLine, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выбор языка.png')

	return (
		<Background src={bgPath}>
			<TextLine x='400px' y='240px' style={{fontSize: '28pt', color: 'white'}}>Windows 7</TextLine>
			<TextLine left='283px' top='274px' style={{fontSize: '9pt'}}>My language is English</TextLine>
			<TextLine left='283px' top='290px' style={{fontSize: '9pt'}}>Мой язык - русский</TextLine>
			<TextLine left='120px' top='490px' style={{fontSize: '8pt', color: 'white'}}>Copyright 2009 Microsoft Corporation. All rights reserved.</TextLine>
		</Background>
	)
}
