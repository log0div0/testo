
import React, { useCallback } from 'react'
import { useDispatch, useSelector } from 'react-redux'
const { dialog } = require('electron').remote
import path from 'path'
import fs from 'fs'
import Select from 'react-select'

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

function TagProps({obj_id, obj}) {
	let dataset_meta = useSelector(state => state.dataset_meta)

	const options = []

	let selected_option = null

	if (dataset_meta.tags) {
		for (let tag of dataset_meta.tags) {
			let option = {
				value: tag,
				label: tag
			}
			if (tag == obj.tag) {
				selected_option = option
			}
			options.push(option)
		}
	}

	let dispatch = useDispatch()

	const styles = {
		container: (provided, state) => {
			return {
				...provided,
				width: "300px"
			}
		}
	}

	let onChange = (option) => {
		dispatch({
			type: 'OBJ_SET_PROP',
			id: obj_id,
			prop: "tag",
			value: option ? option.value : ""
		})
		dispatch({
			type: 'OBJ_UNSELECT'
		})
	}

	return (
		<div id="obj-props-bar">
			<style jsx>{`
				#obj-props-bar {
					padding: 0 12px;
					display: flex;
					align-items: center;
					justify-content: center;
				}
			`}</style>
			<Select value={selected_option} menuPlacement="auto" isClearable={true} styles={styles} options={options} onChange={onChange} />
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
	} if (obj.type == 'tag') {
		return <TagProps obj_id={obj_id} obj={obj}/>
	} else {
		return null
	}
}

export default ObjPropsBar
