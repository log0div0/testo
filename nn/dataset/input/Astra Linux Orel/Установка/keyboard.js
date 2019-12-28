
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Row, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'keyboard.png')

	return (
		<Background src={bgPath}>
			<TextLine x='222px' y='37px' style={{color: 'black', fontSize: '16pt'}}>ASTRALINUX</TextLine>
			<TextLine x='222px' y='55px' style={{color: 'black', fontSize: '10pt'}}>common edition</TextLine>
			<Column right='125px' top='8px'>
				<TextLine style={{fontSize: '11pt', marginBottom: '-3px'}}>Операционная система</TextLine>
				<TextLine style={{fontSize: '11pt'}}>общего назначения</TextLine>
				<TextLine style={{fontSize: '11pt', fontWeight: 'bold'}}>Релиз "Орёл"</TextLine>
			</Column>
			<TextLine left='18px' y='95px' style={{color: 'black', fontSize: '9pt', fontWeight: 'bold'}}>Настройка клавиатуры</TextLine>
			<Column left='16px' top='125px'>
				<TextLine style={{color: 'black', fontSize: '9pt'}}>Вам нужно указать способ переключения клавиатуры между национальной раскладкой и стандартной латинской раскладкой.</TextLine>
			</Column>
			<Column left='16px' top='155px'>
				<TextLine style={{color: 'black', fontSize: '9pt'}}>Наиболее эргономичным способом считаются правая клавиша Alt или Caps Lock (в последнем случае для переключения между заглавнымм и строчными</TextLine>
				<TextLine style={{color: 'black', fontSize: '9pt'}}>буквами используется комюинация Shift+Caps Lock). Ещё одна популярная комбинация: Alt+Shift, заметим, что в этом случае комбинация Alt+Shift потеряет</TextLine>
				<TextLine style={{color: 'black', fontSize: '9pt'}}>своё привычное действие в Emacs и других, использующих её, программах.</TextLine>
			</Column>
			<Column left='16px' top='215px'>
				<TextLine style={{color: 'black', fontSize: '9pt'}}>Не на всех клавиатурах есть перечисленные клавиши.</TextLine>
			</Column>
			<TextLine left='23px' y='243px' style={{color: 'black', fontSize: '9pt', fontStyle: 'italic'}}>Способ переключения между национальной и латинской раскладкой:</TextLine>

			<TextLine left='20px' top='268px' style={{color: 'black', fontSize: '9pt'}}>Caps Lock</TextLine>
			<TextLine left='20px' top='291px' style={{color: 'black', fontSize: '9pt'}}>правый Alt (AltGr)</TextLine>
			<TextLine left='20px' top='314px' style={{color: 'black', fontSize: '9pt'}}>правый Control</TextLine>
			<TextLine left='20px' top='337px' style={{color: 'black', fontSize: '9pt'}}>правый Shift</TextLine>
			<TextLine left='20px' top='360px' style={{color: 'black', fontSize: '9pt'}}>правая клавиша с логотипом</TextLine>
			<TextLine left='20px' top='383px' style={{color: 'black', fontSize: '9pt'}}>клавиша с меню</TextLine>
			<TextLine left='20px' top='406px' style={{color: 'white', fontSize: '9pt'}}>Alt+Shift</TextLine>
			<TextLine left='20px' top='429px' style={{color: 'black', fontSize: '9pt'}}>Control+Shift</TextLine>
			<TextLine left='20px' top='452px' style={{color: 'black', fontSize: '9pt'}}>Control+Alt</TextLine>
			<TextLine left='20px' top='475px' style={{color: 'black', fontSize: '9pt'}}>Alt+Caps Lock</TextLine>
			<TextLine left='20px' top='498px' style={{color: 'black', fontSize: '9pt'}}>левый Control+левый Shift</TextLine>
			<TextLine left='20px' top='521px' style={{color: 'black', fontSize: '9pt'}}>левый Alt</TextLine>
			<TextLine left='20px' top='544px' style={{color: 'black', fontSize: '9pt'}}>левый Control</TextLine>
			<TextLine left='20px' top='567px' style={{color: 'black', fontSize: '9pt'}}>левый Shift</TextLine>
			<TextLine left='20px' top='590px' style={{color: 'black', fontSize: '9pt'}}>левая клавиша с логотипом</TextLine>
			<TextLine left='20px' top='613px' style={{color: 'black', fontSize: '9pt'}}>Scroll Lock</TextLine>
			<TextLine left='20px' top='636px' style={{color: 'black', fontSize: '9pt'}}>без переключателя</TextLine>

			<TextLine x='68px' y='742px' style={{color: 'black', fontSize: '9pt'}}>Снимок экрана</TextLine>
			<TextLine x='180px' y='742px' style={{color: 'black', fontSize: '9pt'}}>Справка</TextLine>
			<TextLine x='840px' y='742px' style={{color: 'black', fontSize: '9pt'}}>Вернуться</TextLine>
			<TextLine x='958px' y='742px' style={{color: 'black', fontSize: '9pt'}}>Продолжить</TextLine>
		</Background>
	)
}
