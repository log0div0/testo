
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Ход установки.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '12pt', color: '#003399'}}>Установка Windows...</TextLine>
			<TextLine left='128px' top='130px' style={{fontSize: '9pt'}}>Получена вся необходимая информация. Во время установки компьютер будет несколько раз</TextLine>
			<TextLine left='128px' top='145px' style={{fontSize: '9pt'}}>перезагружен.</TextLine>
			<TextLine left='150px' top='188px' style={{fontSize: '9pt', color: 'gray'}}>Копирование файлов Windows</TextLine>
			<TextLine left='150px' top='208px' style={{fontSize: '9pt', fontWeight: 'bold'}}>Распаковка файлов Windows (0%)</TextLine>
			<TextLine left='150px' top='228px' style={{fontSize: '9pt', color: 'gray'}}>Установка компонентов</TextLine>
			<TextLine left='150px' top='248px' style={{fontSize: '9pt', color: 'gray'}}>Установка обновлений</TextLine>
			<TextLine left='150px' top='268px' style={{fontSize: '9pt', color: 'gray'}}>Завершение установки</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
