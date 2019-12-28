
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выберите тип установки.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '12pt', color: '#003399'}}>Выберите тип установки.</TextLine>
			<TextLine left='200px' top='134px' style={{fontSize: '10pt', color: '#0066cc'}}>Обновление</TextLine>
			<TextLine left='200px' top='150px' style={{fontSize: '9pt', color: '#0066cc'}}>Обновление Windows позволит сохранить файлы, параметры и программы. Эта</TextLine>
			<TextLine left='200px' top='165px' style={{fontSize: '9pt', color: '#0066cc'}}>возможность доступна только в том случае, если запущена существующая</TextLine>
			<TextLine left='200px' top='180px' style={{fontSize: '9pt', color: '#0066cc'}}>версия Windows. Перед обновлением рекомендуется архивировать файлы.</TextLine>
			<TextLine left='200px' top='238px' style={{fontSize: '10pt', color: '#0066cc'}}>Полная установка (дополнительные параметры)</TextLine>
			<TextLine left='200px' top='255px' style={{fontSize: '9pt', color: '#0066cc'}}>Установка новой копии Windows. При этом файлы, параметры и программы не</TextLine>
			<TextLine left='200px' top='270px' style={{fontSize: '9pt', color: '#0066cc'}}>будут сохранены. Изменение в организации дисков и разделов доступно только</TextLine>
			<TextLine left='200px' top='285px' style={{fontSize: '9pt', color: '#0066cc'}}>при запуске компьютера с установочного диска. Рекомендуется архивировать</TextLine>
			<TextLine left='200px' top='300px' style={{fontSize: '9pt', color: '#0066cc'}}>файлы до начала установки</TextLine>
			<TextLine left='130px' top='390px' style={{fontSize: '9pt', color: '#0066cc'}}>Помощь в принятии решения</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
