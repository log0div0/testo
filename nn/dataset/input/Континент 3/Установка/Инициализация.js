
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'black screen.png')

	return (
		<Background src={bgPath}>
			<Column left='0px' top='0px'>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Криптографический шлюз с ЦУС "Континент"</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Аппаратная платформа: IPC-VM</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Конфигурация: ЦУС, сервер доступа</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8' style={{marginBottom: '16px'}}>ID: 1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Не обнаружена база данных ЦУС</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Начальная конфигурация ЦУС</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Инициализировать ЦУС с использованием файла конфигурации? (Y/N):  N</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Обнаруженные интерфейсы:</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        Номер   Имя</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        1.      em0</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        2.      em1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        3.      tun0</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Укажите номер внешнего интерфейса: 1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Введите внешний IP адрес шлюза: 192.168.1.1/24</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Продолжить? (Y/N): Y</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Обнаруженные интерфейсы:</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        Номер   Имя</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>        2.      em1</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Укажите номер внутреннего интерфейса. Если их несколько -- того, к которому</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'> подключается АРМ администратора: 2</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Введите внутренний IP адрес шлюза: 192.168.2.1/24</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Продолжить? (Y/N): Y</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Введите адрес маршрутизатора по умолчанию: 192.168.2.254</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Адрес маршрутизатора 192.168.1.254</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Продолжить? (Y/N): Y</ConsoleTextLine>
				<ConsoleTextLine font='Terminus16' color='#a8a8a8'>Использовать внешний носитель для инициализации? (Y/N):  N</ConsoleTextLine>
			</Column>
		</Background>
	)
}
