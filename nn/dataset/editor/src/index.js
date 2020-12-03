
import React  from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
import { createStore } from 'redux'
import rootReducer, {filterAndSortDocs} from './reducers'
import App from './App'
import fs from 'fs'
import path from 'path'


// -------------------------------------------------------------------
// Inspect Element
// -------------------------------------------------------------------

const { remote } = require('electron')
const { Menu, MenuItem } = remote

let rightClickPosition = null

const menu = new Menu()
const menuItem = new MenuItem({
	label: 'Inspect Element',
	click: () => {
		remote.getCurrentWindow().inspectElement(rightClickPosition.x, rightClickPosition.y)
	}
})
menu.append(menuItem)

window.addEventListener('contextmenu', (e) => {
	e.preventDefault()
	rightClickPosition = {x: e.x, y: e.y}
	menu.popup(remote.getCurrentWindow())
}, false)

// -------------------------------------------------------------------

window.DATASET_DIR = remote.process.argv[2]

async function fsListener(event, filename) {
	let parsed = path.parse(filename)
	if ((parsed.ext.toLowerCase() != '.png') &&
		(parsed.ext.toLowerCase() != '.json')) {
		return
	}
	let img_path = path.join(DATASET_DIR, parsed.name + '.png')
	let metadata_path = path.join(DATASET_DIR, parsed.name + '.json')
	if (fs.existsSync(img_path)) {
		let metadata = {
			"objs": {}
		}
		if (fs.existsSync(metadata_path)) {
			let data = await fs.promises.readFile(metadata_path)
			if (!data.length) {
				return
			}
			metadata = JSON.parse(data)
		}
		store.dispatch({
			type: "UPDATE_DOC",
			id: parsed.name,
			metadata
		})
	} else {
		if (fs.existsSync(metadata_path)) {
			fs.promises.unlink(metadata_path)
		}
		store.dispatch({
			type: "REMOVE_DOC",
			id: parsed.name
		})
	}
}

async function main() {
	let docs = {}
	const files = await fs.promises.readdir(DATASET_DIR);
	for (const file of files) {
		let parsed = path.parse(file)
		if (parsed.ext.toLowerCase() != '.png') {
			continue
		}
		let metadata_path = path.join(DATASET_DIR, parsed.name + '.json')
		let metadata = {
			"objs": {}
		}
		if (fs.existsSync(metadata_path)) {
			let data = await fs.promises.readFile(metadata_path)
			metadata = JSON.parse(data)
		}
		docs[parsed.name] = metadata
	}

	let initialState = {
		docs,
		selected_doc: null,
		filters: {
			unverified_only: false
		},

		scale: 1,

		objs_ids: [],
		selected_objs: [],
	}

	filterAndSortDocs(initialState)

	window.store = createStore(rootReducer, initialState)

	fs.watch(DATASET_DIR, fsListener)

	ReactDOM.render(
		<Provider store={store}>
			<App/>
		</Provider>,
		document.getElementById('app')
	)
}

main()
