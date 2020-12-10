
import React from 'react'
import DocList from './DocList'
import Doc from './Doc'
import { useDispatch, useSelector } from 'react-redux'

function App() {
	let selected_doc = useSelector(state => state.selected_doc)
	if (selected_doc) {
		return <Doc/>
	} else {
		return <DocList/>
	}
}

export default App
