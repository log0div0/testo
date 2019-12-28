
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Введите имя пользователя.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='270px' y='207px' style={{fontSize: '18pt', letterSpacing: '-1px'}}><Text style={{letterSpacing: '-2px', fontSize: '26pt'}}>Windows 7</Text> Максимальная</TextLine>
			<TextLine left='150px' top='247px' style={{fontSize: '9pt'}}>Выберите имя пользователя для вашей <Text style={{color: '#0066cc', textDecoration: 'underline'}}>учетной записи</Text>, а также имя компьютера в сети.</TextLine>
			<TextLine left='275px' top='278px' style={{fontSize: '9pt'}}>Введите имя пользователя (например, Андрей):</TextLine>
			<TextLine left='275px' top='320px' style={{fontSize: '9pt'}}>Введите <Text style={{color: '#0066cc', textDecoration: 'underline'}}>имя компьютера:</Text></TextLine>
			<TextLine right='111px' top='484px' style={{fontSize: '9pt'}}>Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.</TextLine>
			<TextLine x='653px' y='523px' style={{fontSize: '9pt', color: '#838383'}}>Далее</TextLine>
		</Background>
	)
}
