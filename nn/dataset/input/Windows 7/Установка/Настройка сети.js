
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Настройка сети.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='142px' top='130px' style={{fontSize: '12pt', color: '#003399'}}>Выберите текущее место расположения компьютера</TextLine>
			<TextLine left='142px' top='170px' style={{fontSize: '9pt'}}>Этот компьютер подключен к сети. Будут автоматически использованы правильные</TextLine>
			<TextLine left='142px' top='185px' style={{fontSize: '9pt'}}>сетевые параметры для этого сетевого размещения.</TextLine>
			<TextLine left='205px' top='220px' style={{fontSize: '11pt', color: '#0066cc'}}>Домашняя сеть</TextLine>
			<TextLine left='205px' top='240px' style={{fontSize: '9pt', color: '#0066cc'}}>Если все компьютеры этой сети находятся у вас дома и известны вам, то такая сеть</TextLine>
			<TextLine left='205px' top='255px' style={{fontSize: '9pt', color: '#0066cc'}}>считается доверенной домашней. Не выбирайте данную сеть, если вы находитесь в</TextLine>
			<TextLine left='205px' top='270px' style={{fontSize: '9pt', color: '#0066cc'}}>таких общественных местах, как кафе и аэропорт.</TextLine>
			<TextLine left='205px' top='305px' style={{fontSize: '11pt', color: '#0066cc'}}>Рабочая сеть</TextLine>
			<TextLine left='205px' top='325px' style={{fontSize: '9pt', color: '#0066cc'}}>Если все компьютеры этой сети вам известны и располагаются на вашей работе, то</TextLine>
			<TextLine left='205px' top='340px' style={{fontSize: '9pt', color: '#0066cc'}}>такая сеть считается доверенной рабочей сетью. Не выбирайте данную сеть, если вы</TextLine>
			<TextLine left='205px' top='355px' style={{fontSize: '9pt', color: '#0066cc'}}>находитесь в таких общественных местах, как кафе и аэропорт.</TextLine>
			<TextLine left='205px' top='390px' style={{fontSize: '11pt', color: '#0066cc'}}>Общественная сеть</TextLine>
			<TextLine left='205px' top='410px' style={{fontSize: '9pt', color: '#0066cc'}}>Если не все компьютеры вам известны (если вы находитесь в общественных местах</TextLine>
			<TextLine left='205px' top='425px' style={{fontSize: '9pt', color: '#0066cc'}}>или подключены к широкополосной сети с мобильного телефона), то такая сеть</TextLine>
			<TextLine left='205px' top='440px' style={{fontSize: '9pt', color: '#0066cc'}}>считается общественной (доверие к таким сетям отсутствует).</TextLine>
			<TextLine left='142px' top='475px' style={{fontSize: '9pt'}}>Если не уверены, выбирайте общественную сеть.</TextLine>
		</Background>
	)
}
