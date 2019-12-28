
import React from 'react'
import path from 'path'
import {TextLine, ConsoleTextLine, ConsoleText, Background, Column} from '../../../common'

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'Выбор локации.png')

	return (
		<Background src={bgPath}>
			<ConsoleTextLine font='Fixed16' color='red' left='295px' top='48px'>[!!] Select your location</ConsoleTextLine>
			<Column left='30px' top='80px'>
				<ConsoleTextLine font='Fixed16' color='black'>The selected location will be used to set your time zone and also for example to help</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black' style={{marginBottom: '15px'}}>select the system locale. Normally this should be the country where you live.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>This is a shortlist of locations based on the language you selected. Choose "other" if</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black' style={{marginBottom: '15px'}}>your location is not listed.</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Country, territory or area:</ConsoleTextLine>
			</Column>
			<Column left='304px' top='208px'>
				<ConsoleTextLine font='Fixed16' color='black'>Antigua and Barbuda</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Australia</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Botswana</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Canada</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Hong Kong</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>India</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Ireland</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>New Zealand</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Nigeria</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Philippines</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Singapore</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>South Africa</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>United Kingdom</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='#e0e0e0'>United States</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Zambia</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>Zimbabwe</ConsoleTextLine>
				<ConsoleTextLine font='Fixed16' color='black'>other</ConsoleTextLine>
			</Column>
			<ConsoleTextLine font='Fixed16' color='black' left='50px' top='490px'>{"<Go back>"}</ConsoleTextLine>
			<ConsoleTextLine font='Fixed16' color='white' left='0px' bottom='8px'>{"<Tab> moves; <Space> selects; <Enter> activates buttons"}</ConsoleTextLine>
		</Background>
	)
}
