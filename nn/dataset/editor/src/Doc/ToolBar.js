
import React from 'react'
import { useDispatch, useSelector } from 'react-redux'

function NewObjTool({type}) {
	let dispatch = useDispatch()
	let onClick = () => dispatch({
		type: "TOGGLE_NEW_OBJ_TOOL",
		new_obj_tool: type
	})
	let new_obj_tool = useSelector(state => state.new_obj_tool)
	return (<>
		<style jsx>{`
			button + button {
				margin-left: 10px;
			}
			.active {
				background-color: #c9a42a;
			}
		`}</style>
		<button onClick={onClick} className={new_obj_tool == type ? "active" : null}>Add {type}</button>
	</>)
}

function VerifyBytton() {
	let dispatch = useDispatch()
	let onClick = () => dispatch({
		type: "TOGGLE_DOC_VERIFY"
	})
	let verified = useSelector(state => {
		if (!state.selected_doc) {
			return false
		}
		return state.docs[state.selected_doc].verified
	})
	if (verified) {
		var text = 'Verified'
		var style_str = `
			background-color: #46b859;
			color: white;
			border: 1px solid white;
		`
	} else {
		var text = 'Verify'
		var style_str = ''
	}
	return (<>
		<style jsx>{`
			button {
				${style_str}
				margin: 0 10px;
			}
		`}</style>
		<button onClick={onClick}>{text}</button>
	</>)
}

function ToolBar() {
	let dispatch = useDispatch()

	let closeDoc = () => dispatch({
		type: "CLOSE_DOC"
	})

	let nextDoc = () => dispatch({
		type: "NEXT_DOC"
	})

	let prevDoc = () => dispatch({
		type: "PREV_DOC"
	})

	return (
		<div id="tool-bar">
			<style jsx>{`
				#tool-bar {
					display: flex;
					justify-content: space-around;
				}
				button + button {
					margin-left: 10px;
				}
				section {
					width: 32%;
					display: flex;
					align-items: center;
				}
				.left {
					justify-content: flex-start;
				}
				.center {
					justify-content: center;
				}
				.right {
					justify-content: flex-end;
				}
			`}</style>
			<section className="left">
				<button onClick={closeDoc}>Close</button>
			</section>
			<section className="center">
				<NewObjTool type="text"/>
			</section>
			<section className="right">
				<button onClick={prevDoc}>Go to prev</button>
				<VerifyBytton/>
				<button onClick={nextDoc}>Go to next</button>
			</section>
		</div>
	)
}

export default ToolBar
