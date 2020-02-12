
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выберите раздел для установки.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '12pt', color: '#003399'}}>Выберите раздел для установки Windows</TextLine>
			<TextLine left='175px' top='141px' style={{fontSize: '9pt'}}>Файл</TextLine>
			<TextLine left='400px' top='141px' style={{fontSize: '9pt'}}>Полный раз...</TextLine>
			<TextLine right='231px' top='141px' style={{fontSize: '9pt'}}>Свободно</TextLine>
			<TextLine left='580px' top='141px' style={{fontSize: '9pt'}}>Тип</TextLine>
			<TextLine left='175px' top='169px' style={{fontSize: '9pt'}}>Незанятое место на диске 0</TextLine>
			<TextLine right='323px' top='169px' style={{fontSize: '9pt'}}>20.0 ГБ</TextLine>
			<TextLine right='230px' top='169px' style={{fontSize: '9pt'}}>20.0 ГБ</TextLine>
			<TextLine left='149px' top='343px' style={{fontSize: '9pt', color: '#0066cc'}}>Обновить</TextLine>
			<TextLine left='149px' top='370px' style={{fontSize: '9pt', color: '#0066cc'}}>Загрузка</TextLine>
			<TextLine left='500px' top='343px' style={{fontSize: '9pt', color: '#0066cc'}}>Настройка диска</TextLine>
			<TextLine x='659px' y='472px' style={{fontSize: '9pt'}}>Далее</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
