
import React from 'react'
import path from 'path'
import {TextLine, Text, Background} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Ознакомьтесь с лицензией.png')

	return (
		<Background src={bgPath}>
			<TextLine left='143px' top='36px' style={{fontSize: '9pt', color: '#000260'}}>Установка Windows</TextLine>
			<TextLine left='128px' top='85px' style={{fontSize: '12pt', color: '#003399'}}>Ознакомьтесь с условиями лицензии</TextLine>
			<TextLine left='135px' top='138px' style={{fontSize: '13px', fontWeight: 'bold'}}>УСЛОВИЯ ЛИЦЕНЗИИ НА ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ MICROSOFT</TextLine>
			<TextLine left='135px' top='168px' style={{fontSize: '13px', fontWeight: 'bold'}}>WINDOWS 7 МАКСИМАЛЬНАЯ С ПАКЕТОМ ОБНОВЛЕНИЯ 1</TextLine>
			<TextLine left='135px' top='198px' style={{fontSize: '13px'}}>Настоящие условия лицензии являются соглашением между корпорацией Microsoft</TextLine>
			<TextLine left='135px' top='213px' style={{fontSize: '13px'}}>(или, в зависимости от места вашего проживания, одним из её аффилированных</TextLine>
			<TextLine left='135px' top='228px' style={{fontSize: '13px'}}>лиц) и вами. Прочтите их внимательно. Они применяются к вышеуказанному</TextLine>
			<TextLine left='135px' top='243px' style={{fontSize: '13px'}}>программному обеспечению, включая носители, на которых оно распространяется</TextLine>
			<TextLine left='135px' top='258px' style={{fontSize: '13px'}}>(если они есть). Условия лицензионного соглашения, представляемые в печатном</TextLine>
			<TextLine left='135px' top='273px' style={{fontSize: '13px'}}>виде, которые могу сопровождать программное обеспечение, имеют</TextLine>
			<TextLine left='135px' top='288px' style={{fontSize: '13px'}}>преимущественную силу над любыми условиями лицензии, представляемыми в</TextLine>
			<TextLine left='135px' top='303px' style={{fontSize: '13px'}}>электронном виде. Эти условия распространяются также на все</TextLine>
			<TextLine left='150px' top='338px' style={{fontSize: '13px'}}>обновления,</TextLine>
			<TextLine left='150px' top='373px' style={{fontSize: '13px'}}>дополнительные компоненты,</TextLine>
			<TextLine left='146px' top='418px' style={{fontSize: '9pt'}}>Я принимаю условия лицензии</TextLine>
			<TextLine x='658px' y='472px' style={{fontSize: '9pt', color: 'grey'}}>Далее</TextLine>
			<TextLine left='10px' top='551px' style={{fontSize: '26pt', color: 'white'}}>1</TextLine>
			<TextLine left='40px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Сбор информации</TextLine>
			<TextLine left='200px' top='551px' style={{fontSize: '26pt', color: 'white'}}>2</TextLine>
			<TextLine left='230px' top='557px' style={{fontSize: '9pt', color: 'white'}}>Установка Windows</TextLine>
		</Background>
	)
}
