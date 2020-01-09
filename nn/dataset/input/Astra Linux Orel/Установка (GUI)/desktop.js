
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Row, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'desktop.png')

	return (
		<Background src={bgPath}>
			<Column x='53px' top='60px'>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt', marginBottom: '4px'}}>Веб-браузер</TextLine>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt'}}>Firefox</TextLine>
			</Column>
			<Column x='53px' top='150px'>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt'}}>Корзина</TextLine>
			</Column>
			<Column x='53px' top='244px'>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt'}}>Мой</TextLine>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt'}}>компьютер</TextLine>
			</Column>
			<Column x='53px' top='340px'>
				<TextLine style={{color: 'white', textShadow: '1px 1px #5a5a5a', fontSize: '10pt'}}>Помощь</TextLine>
			</Column>
			<TextLine x='850px' y='640px' style={{color: 'white', fontSize: '20pt'}}>ASTRALINUX</TextLine>
			<TextLine x='219px' y='736px' style={{color: 'black', fontSize: '9pt'}}>1</TextLine>
			<TextLine x='250px' y='736px' style={{color: 'white', fontSize: '9pt'}}>2</TextLine>
			<TextLine x='219px' y='759px' style={{color: 'white', fontSize: '9pt'}}>3</TextLine>
			<TextLine x='250px' y='759px' style={{color: 'white', fontSize: '9pt'}}>4</TextLine>
			<TextLine x='918px' y='742px' style={{color: 'white', fontSize: '12pt'}}>EN</TextLine>
			<TextLine x='978px' y='732px' style={{color: 'white', fontSize: '14pt'}}>15:00</TextLine>
			<TextLine x='978px' y='752px' style={{color: 'white', fontSize: '12pt'}}>Ср, 18 дек</TextLine>
		</Background>
	)
}
