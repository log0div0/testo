
import React from 'react'
import path from 'path'
import {TextLine, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выбор языка 2.png')

	return (
		<Background src={bgPath}>
			<TextLine left='115px' top='75px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
			<TextLine x='400px' y='240px' style={{fontSize: '28pt', color: 'white'}}>Windows 7</TextLine>
			<TextLine right='450px' top='300px' style={{fontSize: '9pt', color: 'white'}}>Устанавливаемый язык:</TextLine>
			<TextLine right='450px' top='337px' style={{fontSize: '9pt', color: 'white'}}>Формат времени и денежных единиц:</TextLine>
			<TextLine right='450px' top='373px' style={{fontSize: '9pt', color: 'white'}}>Раскладка клавиатуры или метод ввода:</TextLine>
			<TextLine left='360px' top='300px' style={{fontSize: '9pt', color: 'white'}}>Русский</TextLine>
			<TextLine left='360px' top='337px' style={{fontSize: '9pt', color: 'black'}}>Русский (Россия)</TextLine>
			<TextLine left='360px' top='373px' style={{fontSize: '9pt', color: 'black'}}>Русская</TextLine>
			<TextLine x='400px' y='448px' style={{fontSize: '9pt', color: 'white'}}>Введите нужный язык и другие параметры, а затем нажмите кнопку "Далее".</TextLine>
			<TextLine left='120px' top='480px' style={{fontSize: '8pt', color: 'white'}}>Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.</TextLine>
			<TextLine x='660px' y='498px' style={{fontSize: '9pt', color: 'black'}}>Далее</TextLine>
		</Background>
	)
}
