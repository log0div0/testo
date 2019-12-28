
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Введите пароль.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='143px' top='130px' style={{fontSize: '12pt', color: '#003399'}}>Установите пароль для своей учетной записи</TextLine>
			<TextLine left='143px' top='170px' style={{fontSize: '9pt'}}>Создание пароля - это разумная мера предосторожности для защиты вашей учетной</TextLine>
			<TextLine left='143px' top='185px' style={{fontSize: '9pt'}}>записи от нежелательного использования. Запомните пароль или храните его в надёжном</TextLine>
			<TextLine left='143px' top='200px' style={{fontSize: '9pt'}}>месте.</TextLine>
			<TextLine left='143px' top='232px' style={{fontSize: '9pt'}}>Введите пароль (рекомендуется):</TextLine>
			<TextLine left='143px' top='282px' style={{fontSize: '9pt'}}>Подтверждение пароля:</TextLine>
			<TextLine left='143px' top='332px' style={{fontSize: '9pt'}}>Введите подсказку для пароля:</TextLine>
			<TextLine left='143px' top='380px' style={{fontSize: '9pt'}}>Выберите слово или фразу, с помощью которых можно лучше запомнить пароль.</TextLine>
			<TextLine left='143px' top='395px' style={{fontSize: '9pt'}}>Если вы забудете пароль, на экране появится введенная подсказка.</TextLine>
			<TextLine x='653px' y='523px' style={{fontSize: '9pt'}}>Далее</TextLine>
		</Background>
	)
}
