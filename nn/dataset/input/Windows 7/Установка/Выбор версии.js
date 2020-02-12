
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выбор версии.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '11pt', color: '#003399'}}>Выберите операционную систему, которую следует установить</TextLine>
			<TextLine left='137px' top='131px' style={{fontSize: '9pt'}}>Операционная система</TextLine>
			<TextLine left='470px' top='131px' style={{fontSize: '9pt'}}>Архитектура</TextLine>
			<TextLine left='565px' top='131px' style={{fontSize: '9pt'}}>Дата измене...</TextLine>
			<TextLine left='137px' top='151px' style={{fontSize: '9pt', color: 'white'}}>Windows 7 Начальная</TextLine>
			<TextLine left='470px' top='151px' style={{fontSize: '9pt', color: 'white'}}>x86</TextLine>
			<TextLine left='570px' top='151px' style={{fontSize: '9pt', color: 'white'}}>11/20/2010</TextLine>
			<TextLine left='137px' top='167px' style={{fontSize: '9pt'}}>Windows 7 Домашняя базовая</TextLine>
			<TextLine left='470px' top='167px' style={{fontSize: '9pt'}}>x86</TextLine>
			<TextLine left='570px' top='167px' style={{fontSize: '9pt'}}>11/20/2010</TextLine>
			<TextLine left='137px' top='183px' style={{fontSize: '9pt'}}>Windows 7 Домашняя базовая</TextLine>
			<TextLine left='470px' top='183px' style={{fontSize: '9pt'}}>x64</TextLine>
			<TextLine left='570px' top='183px' style={{fontSize: '9pt'}}>11/21/2010</TextLine>
			<TextLine left='137px' top='199px' style={{fontSize: '9pt'}}>Windows 7 Домашняя расширенная</TextLine>
			<TextLine left='470px' top='199px' style={{fontSize: '9pt'}}>x86</TextLine>
			<TextLine left='570px' top='199px' style={{fontSize: '9pt'}}>11/20/2010</TextLine>
			<TextLine left='137px' top='215px' style={{fontSize: '9pt'}}>Windows 7 Домашняя расширенная</TextLine>
			<TextLine left='470px' top='215px' style={{fontSize: '9pt'}}>x64</TextLine>
			<TextLine left='570px' top='215px' style={{fontSize: '9pt'}}>11/21/2010</TextLine>
			<TextLine left='137px' top='232px' style={{fontSize: '9pt'}}>Windows 7 Профессиональная</TextLine>
			<TextLine left='470px' top='232px' style={{fontSize: '9pt'}}>x86</TextLine>
			<TextLine left='570px' top='232px' style={{fontSize: '9pt'}}>11/20/2010</TextLine>
			<TextLine left='137px' top='248px' style={{fontSize: '9pt'}}>Windows 7 Профессиональная</TextLine>
			<TextLine left='470px' top='248px' style={{fontSize: '9pt'}}>x64</TextLine>
			<TextLine left='570px' top='248px' style={{fontSize: '9pt'}}>11/21/2010</TextLine>
			<TextLine left='137px' top='264px' style={{fontSize: '9pt'}}>Windows 7 Максимальная</TextLine>
			<TextLine left='470px' top='264px' style={{fontSize: '9pt'}}>x86</TextLine>
			<TextLine left='570px' top='264px' style={{fontSize: '9pt'}}>11/20/2010</TextLine>
			<TextLine left='137px' top='280px' style={{fontSize: '9pt'}}>Windows 7 Максимальная</TextLine>
			<TextLine left='470px' top='280px' style={{fontSize: '9pt'}}>x64</TextLine>
			<TextLine left='570px' top='280px' style={{fontSize: '9pt'}}>11/21/2010</TextLine>
			<TextLine left='137px' top='300px' style={{fontSize: '9pt'}}>Описание:</TextLine>
			<TextLine left='137px' top='315px' style={{fontSize: '9pt'}}>Windows 7 Начальная</TextLine>
			<TextLine x='659px' y='472px' style={{fontSize: '9pt'}}>Далее</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
