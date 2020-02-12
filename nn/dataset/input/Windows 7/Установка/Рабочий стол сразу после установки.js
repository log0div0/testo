
import React from 'react'
import path from 'path'
import {TextLine, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Рабочий стол сразу после установки.png')

	return (
		<Background src={bgPath}>
			<TextLine x='40px' y='65px' style={{fontSize: '9pt', color: 'white', textShadow: '1px 2px 2px black'}}>Корзина</TextLine>
			<TextLine x='620px' y='580px' style={{fontSize: '10pt', color: 'white'}}>RU</TextLine>
			<TextLine x='743px' y='571px' style={{fontSize: '9pt', color: 'white'}}>13:56</TextLine>
			<TextLine x='743px' y='587px' style={{fontSize: '9pt', color: 'white'}}>23.11.2019</TextLine>
		</Background>
	)
}
