
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Помогите автоматически защитить компьютер.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='142px' top='130px' style={{fontSize: '12pt', color: '#003399'}}>Помогите автоматически защитить компьютер и улучшить Windows</TextLine>
			<TextLine left='200px' top='180px' style={{fontSize: '11pt', color: '#251f6f'}}>Использовать рекомендуемые параметры</TextLine>
			<TextLine left='200px' top='200px' style={{fontSize: '9pt', color: '#251f6f'}}>Установка важных и рекомендуемыx обвновлений, обеспечение более безопасного</TextLine>
			<TextLine left='200px' top='215px' style={{fontSize: '9pt', color: '#251f6f'}}>обзора Интернета, поиск решений для возникающих проблем в Интернете и</TextLine>
			<TextLine left='200px' top='230px' style={{fontSize: '9pt', color: '#251f6f'}}>помощь в улучшении Microsoft Windows.</TextLine>
			<TextLine left='200px' top='270px' style={{fontSize: '11pt', color: '#251f6f'}}>Устанавливать только наиболее важные обновления</TextLine>
			<TextLine left='200px' top='290px' style={{fontSize: '9pt', color: '#251f6f'}}>Устанавливать только обновления безопасности и другие наиболее важные</TextLine>
			<TextLine left='200px' top='305px' style={{fontSize: '9pt', color: '#251f6f'}}>обвновления для Windows.</TextLine>
			<TextLine left='200px' top='340px' style={{fontSize: '11pt', color: '#251f6f'}}>Отложить решение</TextLine>
			<TextLine left='200px' top='360px' style={{fontSize: '9pt', color: '#251f6f'}}>Пока решение не будет принято, безопасность компьютера остаётся под угрозой.</TextLine>
			<TextLine left='142px' top='400px' style={{fontSize: '9pt', color: 'blue', textDecoration: 'underline'}}>Подробнее об этих параметрах</TextLine>
			<TextLine left='142px' top='420px' style={{fontSize: '9pt'}}>При использовании рекомендуемыx параметров или при установке обвновлений некоторые</TextLine>
			<TextLine left='142px' top='435px' style={{fontSize: '9pt'}}>сведения передаются в корпорацию Microsoft. Они не используются с целью</TextLine>
			<TextLine left='142px' top='450px' style={{fontSize: '9pt'}}>установления личности пользователя или связи с ним. Чтобы отключить эти параметры</TextLine>
			<TextLine left='142px' top='465px' style={{fontSize: '9pt'}}>позднее, выполните поиск по словам "отключение рекомендуемыx параметров" в центре</TextLine>
			<TextLine left='142px' top='480px' style={{fontSize: '9pt'}}>справки и поддержки. <Text style={{color: 'blue', textDecoration: 'underline'}}>Заявление о конфиденциальности</Text></TextLine>
		</Background>
	)
}
