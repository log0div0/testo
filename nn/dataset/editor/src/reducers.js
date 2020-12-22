
import produce from 'immer'
import { v4 as uuidv4 } from 'uuid'
import fs from 'fs'
import path from 'path'

function toggleNewObjTool(state, new_obj_tool) {
	if (state.new_obj_tool == new_obj_tool) {
		state.new_obj_tool = null
	} else {
		state.new_obj_tool = new_obj_tool
	}
}

const newObjToolMap = {
	Digit1: 'text',
	Digit2: 'tag',
}

function keyboardReducer(state, action) {
	if (document.activeElement instanceof HTMLInputElement) {
		return
	}
	switch (action.code) {
		case 'Digit1':
		case 'Digit2':
			if (!action.isDown) {
				return
			}
			toggleNewObjTool(state, newObjToolMap[action.code])
			break
		case 'KeyD':
			if (!action.isDown) {
				return
			}
			if (!action.ctrlKey) {
				return
			}
			let new_selected_objs = []
			for (let obj_id of state.selected_objs) {
				let obj = state.docs[state.selected_doc].objs[obj_id]
				let new_obj = JSON.parse(JSON.stringify(obj))

				let new_obj_id = uuidv4()
				state.docs[state.selected_doc].objs[new_obj_id] = new_obj
				state.objs_ids.push(new_obj_id)
				new_selected_objs.push(new_obj_id)
			}
			state.selected_objs = new_selected_objs
			break
		case 'Delete':
			if (!action.isDown) {
				return
			}
			if (!state.selected_objs.length) {
				return
			}
			let doc = state.docs[state.selected_doc]
			for (let id of state.selected_objs) {
				let deleted_obj = doc.objs[id]
				delete doc.objs[id]
				var index = state.objs_ids.indexOf(id);
				if (index >= 0) {
					state.objs_ids.splice(index, 1);
				}
			}
			state.selected_objs = []
			break
	}
}

function mouseReducer(state, action) {
	switch (action.code) {
		case 'mousedown':
			if (state.selected_objs.length != 0) {
				if ((action.target_name == 'stage') ||
					(action.target_name == 'background'))
				{
					state.selected_objs = []
				}
			}
			if (state.new_obj_tool) {
				let new_obj = {
					type: state.new_obj_tool,
					x: action.x,
					y: action.y,
					width: 0,
					height: 0
				}
				if (state.new_obj_tool == 'text') {
					new_obj.text = ""
				} else if (state.new_obj_tool == 'tag') {
					new_obj.tag = ""
				}
				let obj_id = uuidv4()
				state.docs[state.selected_doc].objs[obj_id] = new_obj
				state.selected_objs = [obj_id]
				state.objs_ids.push(obj_id)
				state.new_obj = {
					id: obj_id,
					x: action.x,
					y: action.y
				}
			}
			break
		case 'mousemove':
			if (state.new_obj) {
				let obj = state.docs[state.selected_doc].objs[state.new_obj.id]
				let x1 = Math.min(state.new_obj.x, action.x)
				let y1 = Math.min(state.new_obj.y, action.y)
				let x2 = Math.max(state.new_obj.x, action.x)
				let y2 = Math.max(state.new_obj.y, action.y)
				obj.x = x1
				obj.y = y1
				obj.width = x2 - x1
				obj.height = y2 - y1
			}
			break
		case 'mouseup':
			if (state.new_obj) {
				state.new_obj_tool = null
				state.new_obj = null
			}
			break
	}
}

export function filterAndSortDocs(state) {
	let docs_ids = []
	for (let id in state.docs) {
		if (state.filters.unverified_only) {
			let doc = state.docs[id]
			if (!doc.verified) {
				docs_ids.push(id)
			}
		} else {
			docs_ids.push(id)
		}
	}
	docs_ids.sort()
	state.docs_ids = docs_ids
}

function saveDoc(state) {
	let doc = state.docs[state.selected_doc]
	let metadata = JSON.stringify(doc, function(key, val) {
		return val.toFixed ? Number(val.toFixed(2)) : val;
	}, '\t')
	let metadata_path = path.join(DATASET_DIR, state.selected_doc + '.json')
	fs.writeFileSync(metadata_path, metadata, 'utf8')
}

function openDoc(state, docId) {
	document.title = docId
	state.new_obj_tool = null
	state.selected_doc = docId
	state.selected_objs = []
	state.objs_ids = Object.keys(state.docs[docId].objs)
}

function closeDoc(state) {
	state.new_obj_tool = null
	state.selected_doc = null
	state.selected_objs = []
	state.objs_ids = []
	filterAndSortDocs(state)
	document.title = "dataset-editor"
}

function rootReducer(state, action) {
	switch (action.type) {
		case 'SET_SHOW_META':
			state.show_meta = action.value
			break
		case 'TOGGLE_NEW_OBJ_TOOL':
			toggleNewObjTool(state, action.new_obj_tool)
			break
		case 'TOGGLE_DOC_VERIFY':
			if (state.docs[state.selected_doc].verified) {
				state.docs[state.selected_doc].verified = false
			} else {
				state.docs[state.selected_doc].verified = true
			}
			break
		case 'CLICK_OBJ':
			if ((state.selected_objs.length == 1) && (state.selected_objs[0] == action.id)) {
				return state
			}
			state.selected_objs = [action.id]
			break
		case 'SET_FILTER':
			state.filters[action.name] = action.value
			filterAndSortDocs(state)
			break
		case 'OBJ_SET_PROP':
			state.docs[state.selected_doc].objs[action.id][action.prop] = action.value
			break
		case 'OBJ_UNSELECT':
			if (state.selected_objs.length) {
				state.selected_objs = []
			}
			break
		case 'TRANSFORM_OBJ':
			let obj = state.docs[state.selected_doc].objs[action.id]
			obj.x = action.x;
			obj.y = action.y;
			obj.width = action.width;
			obj.height = action.height;
			break
		case 'MOUSE_EVENT':
			mouseReducer(state, action)
			break
		case 'KEYBOARD_EVENT':
			keyboardReducer(state, action)
			break
		case 'OPEN_DOC':
			openDoc(state, action.id)
			break
		case 'NEXT_DOC':
			saveDoc(state)
			var index = state.docs_ids.indexOf(state.selected_doc)
			if (index == (state.docs_ids.length - 1)) {
				return closeDoc(state)
			}
			openDoc(state, state.docs_ids[index + 1])
			break
		case 'PREV_DOC':
			saveDoc(state)
			var index = state.docs_ids.indexOf(state.selected_doc)
			if (index == 0) {
				return closeDoc(state)
			}
			openDoc(state, state.docs_ids[index - 1])
			break
		case 'CLOSE_DOC':
			saveDoc(state)
			closeDoc(state)
			break
		case 'UPDATE_DATASET_META':
			state.dataset_meta = action.dataset_meta
			break
		case 'UPDATE_DOC':
			state.docs[action.id] = action.metadata

			if (state.selected_doc == action.id) {
				let selected_objs = []
				for (let id of state.selected_objs) {
					if (id in state.docs[state.selected_doc].objs) {
						selected_objs.push(id)
					}
				}
				state.selected_objs = selected_objs
				state.objs_ids = Object.keys(state.docs[state.selected_doc].objs)
			}

			break
		case 'REMOVE_DOC':
			delete state.docs[action.id]
			var index = state.docs_ids.indexOf(action.id);
			if (index >= 0) {
				state.docs_ids.splice(index, 1);
			}
			if (state.selected_doc == action.id) {
				state.selected_doc = null
				state.selected_objs = []
				state.objs_ids = []
			}
			break
		case 'NEW_SCALE':
			state.scale = action.value
			break
	}
	return state
}

export default produce(rootReducer)
