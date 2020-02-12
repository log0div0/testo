
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Настройка даты и времени.png')

	return (
		<Background src={bgPath}>
			<TextLine x='538px' y='12px' style={{fontSize: '9pt', color: 'white'}}>RU Русский (Россия)</TextLine>
			<TextLine x='649px' y='12px' style={{fontSize: '9pt', color: 'white'}}>Справка</TextLine>
			<TextLine left='163px' y='93px' style={{fontSize: '9pt'}}>Настройка Windows</TextLine>
			<TextLine left='142px' top='130px' style={{fontSize: '12pt', color: '#003399'}}>Проверьте настройку даты и времени</TextLine>
			<TextLine left='142px' top='170px' style={{fontSize: '9pt'}}>Часовой пояс:</TextLine>
			<TextLine left='150px' top='197px' style={{fontSize: '9pt'}}>(UTC+03:00) Волгоград, Москва, Санкт-Петербург</TextLine>
			<TextLine left='161px' top='224px' style={{fontSize: '9pt'}}>Автоматический переход на летнее время и обратно</TextLine>
			<TextLine left='142px' top='260px' style={{fontSize: '9pt'}}>Дата:</TextLine>
			<TextLine left='350px' top='260px' style={{fontSize: '9pt'}}>Время:</TextLine>
			<TextLine left='182px' top='292px' style={{fontSize: '9pt'}}>Ноябрь 2019</TextLine>

			<TextLine left='139px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Пн</TextLine>
			<TextLine left='161px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Вт</TextLine>
			<TextLine left='183px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Ср</TextLine>
			<TextLine left='205px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Чт</TextLine>
			<TextLine left='227px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Пт</TextLine>
			<TextLine left='249px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Сб</TextLine>
			<TextLine left='271px' top='315px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>Вс</TextLine>

			<TextLine left='139px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>28</TextLine>
			<TextLine left='161px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>29</TextLine>
			<TextLine left='183px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>30</TextLine>
			<TextLine left='205px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>31</TextLine>
			<TextLine left='227px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>1</TextLine>
			<TextLine left='249px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>2</TextLine>
			<TextLine left='271px' top='332px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>3</TextLine>

			<TextLine left='139px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>4</TextLine>
			<TextLine left='161px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>5</TextLine>
			<TextLine left='183px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>6</TextLine>
			<TextLine left='205px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>7</TextLine>
			<TextLine left='227px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>8</TextLine>
			<TextLine left='249px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>9</TextLine>
			<TextLine left='271px' top='347px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>10</TextLine>

			<TextLine left='139px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>11</TextLine>
			<TextLine left='161px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>12</TextLine>
			<TextLine left='183px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>13</TextLine>
			<TextLine left='205px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>14</TextLine>
			<TextLine left='227px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>15</TextLine>
			<TextLine left='249px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>16</TextLine>
			<TextLine left='271px' top='362px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>17</TextLine>

			<TextLine left='139px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>18</TextLine>
			<TextLine left='161px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>19</TextLine>
			<TextLine left='183px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>20</TextLine>
			<TextLine left='205px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>21</TextLine>
			<TextLine left='227px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>22</TextLine>
			<TextLine left='249px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: '#003399'}}>23</TextLine>
			<TextLine left='271px' top='377px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>24</TextLine>

			<TextLine left='139px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>25</TextLine>
			<TextLine left='161px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>26</TextLine>
			<TextLine left='183px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>27</TextLine>
			<TextLine left='205px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>28</TextLine>
			<TextLine left='227px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>29</TextLine>
			<TextLine left='249px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px'}}>30</TextLine>
			<TextLine left='271px' top='392px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>1</TextLine>

			<TextLine left='139px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>2</TextLine>
			<TextLine left='161px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>3</TextLine>
			<TextLine left='183px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>4</TextLine>
			<TextLine left='205px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>5</TextLine>
			<TextLine left='227px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>6</TextLine>
			<TextLine left='249px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>7</TextLine>
			<TextLine left='271px' top='407px' style={{fontSize: '9pt', textAlign: 'right', display: 'inline-block', width: '22px', color: 'gray'}}>8</TextLine>

			<TextLine x='394px' y='428px' style={{fontSize: '9pt'}}>13:54:30</TextLine>
			<TextLine x='652px' y='523px' style={{fontSize: '9pt'}}>Далее</TextLine>
		</Background>
	)
}
