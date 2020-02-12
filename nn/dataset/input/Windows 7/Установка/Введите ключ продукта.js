
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Введите ключ продукта.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='140px' top='130px' style={{fontSize: '12pt', color: '#003399'}}>Введите ключ продукта Windows</TextLine>
			<TextLine left='140px' top='170px' style={{fontSize: '9pt'}}>Наклейка с ключом продукта находится либо на упаковке копии Windows, либо на корпусе</TextLine>
			<TextLine left='140px' top='185px' style={{fontSize: '9pt'}}>компьютера. При активации ключ продукта будет привязан к вашему компьютеру.</TextLine>
			<TextLine left='140px' top='215px' style={{fontSize: '9pt'}}>Наклейка с ключом продукта выглядит так:</TextLine>
			<TextLine left='140px' top='237px' style={{fontSize: '9pt'}}>КЛЮЧ ПРОДУКТА: XXXXX-XXXXX-XXXXX-XXXXX-XXXXX</TextLine>
			<TextLine left='140px' top='287px' style={{fontSize: '9pt'}}>(дефисы вводятся автоматически)</TextLine>
			<TextLine left='159px' top='320px' style={{fontSize: '9pt'}}>Автоматически активировать Windows при подключении к Интернету</TextLine>
			<TextLine left='140px' top='430px' style={{fontSize: '9pt', color: '#0066cc', textDecoration: 'underline'}}>Что такое активация?</TextLine>
			<TextLine left='140px' top='447px' style={{fontSize: '9pt', color: '#0066cc', textDecoration: 'underline'}}>Заявление о конфиденциальности</TextLine>
			<TextLine x='563px' y='523px' style={{fontSize: '9pt'}}>Пропустить</TextLine>
			<TextLine x='653px' y='523px' style={{fontSize: '9pt'}}>Далее</TextLine>
		</Background>
	)
}
