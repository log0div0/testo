
import React, { useCallback } from 'react'
import { useDispatch, useSelector } from 'react-redux'
const { dialog } = require('electron').remote
import path from 'path'
import fs from 'fs'

function TextProps({obj_id, obj}) {
	let dispatch = useDispatch()

	let setStrProp = (prop) => {
		return (e) => {
			if (!obj_id) {
				return
			}
			dispatch({
				type: 'OBJ_SET_PROP',
				id: obj_id,
				prop,
				value: e.target.value
			})
		}
	}

	const handleKeyDown = (event) => {
		if (event.key === 'Enter') {
			dispatch({
				type: 'OBJ_UNSELECT'
			})
		}
	}

	return (
		<div id="obj-props-bar">
			<style jsx>{`
				#obj-props-bar {
					padding: 0 12px;
					display: flex;
					align-items: center;
				}
				input {
					width: 100%;
					text-align: center;
					font-size: 18px;
					font-family: Consolas, "Ubuntu Mono", mono;
					padding: 5px 0;
				}
			`}</style>
			<input type="text" value={obj.text ? obj.text : ""} onKeyDown={handleKeyDown} onChange={setStrProp('text')}/>
		</div>
	)
}

function ObjPropsBar() {
	let obj_id = useSelector(state => {
		if (state.selected_objs.length != 1) {
			return null
		}
		return state.selected_objs[0]
	})
	let obj = useSelector(state => {
		if (!state.selected_doc) {
			return null
		}
		if (state.selected_objs.length != 1) {
			return null
		}
		return state.docs[state.selected_doc].objs[state.selected_objs[0]]
	})

	let dispatch = useDispatch()

	if (!obj) {
		return null
	}

	if (obj.type == 'text') {
		return <TextProps obj_id={obj_id} obj={obj}/>
	} else {
		return null
	}
}

export default ObjPropsBar
