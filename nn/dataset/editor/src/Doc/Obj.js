
import React, {useCallback} from 'react'
import { Rect, Text, Image } from 'react-konva'
import { useDispatch, useSelector } from 'react-redux'
import useImage from 'use-image'

let TextObj = React.forwardRef(({obj, draggable, onDragMove, onClick, onTransform}, ref) => {
	let show_meta = useSelector(state => state.show_meta)

	if (obj.text) {
		var stroke = '#0e2f88'
		if (show_meta) {
			var fill = '#a7b4d9'
		} else {
			var fill = '#0e2f8833'
		}
	} else {
		var stroke = '#880e0e'
		if (show_meta) {
			var fill = '#880e0e'
		} else {
			var fill = '#880e0e33'
		}
	}

	return (
		<>
			<Rect
				x={obj.x}
				y={obj.y}
				fill={fill}
				width={obj.width}
				height={obj.height}
				strokeWidth={1}
				scale={{x: 1, y: 1}}
				stroke={stroke}
				draggable={draggable}
				onDragMove={onDragMove}
				onClick={onClick}
				onTransform={onTransform}
				name='obj'
				ref={ref}
			></Rect>
			{ show_meta ? <Text
				x={obj.x + 1}
				y={obj.y + 1}
				height={obj.height - 2}
				text={obj.text}
				fontSize={obj.height / 2}
				fontFamily='mono'
				fill='#555'
				verticalAlign='middle'
			/> : null }
		</>
	)
})

let TagObj = React.forwardRef(({obj, draggable, onDragMove, onClick, onTransform}, ref) => {

	let show_meta = useSelector(state => state.show_meta)

	if (obj.tag) {
		var stroke = '#fcba03'
		var fill = '#fcba0355'
		if (show_meta) {
			var fill = '#fae57f'
		} else {
			var fill = '#fcba0355'
		}
	} else {
		var stroke = '#880e0e'
		if (show_meta) {
			var fill = '#880e0e'
		} else {
			var fill = '#880e0e33'
		}
	}

	return (
		<>
			<Rect
				x={obj.x}
				y={obj.y}
				fill={fill}
				width={obj.width}
				height={obj.height}
				strokeWidth={1}
				scale={{x: 1, y: 1}}
				stroke={stroke}
				draggable={draggable}
				onDragMove={onDragMove}
				onClick={onClick}
				onTransform={onTransform}
				name='obj'
				ref={ref}
			/>
			{ show_meta ? <Text
				x={obj.x + 1}
				y={obj.y + 1}
				height={obj.height - 2}
				width={obj.width - 2}
				text={obj.tag}
				fontSize={obj.width / (obj.tag.length)}
				fontFamily='mono'
				fontStyle='bold'
				fill='#555'
				verticalAlign='middle'
				align='center'
			/> : null }
		</>
	)
})

function Obj({id}, ref) {
	let dispatch = useDispatch()

	let onDragMove = useCallback(e => {
		dispatch({
			type: 'TRANSFORM_OBJ',
			id,
			x: e.target.x(),
			y: e.target.y(),
			width: e.target.width(),
			height: e.target.height(),
		})
	}, [dispatch])

	let onClick = useCallback(e => {
		if (e.evt.button != 0) {
			return
		}
		dispatch({
			type: 'CLICK_OBJ',
			id
		})
	}, [dispatch])

	let onTransform = useCallback(e => {
		dispatch({
			type: 'TRANSFORM_OBJ',
			id,
			x: e.target.x(),
			y: e.target.y(),
			width: e.target.width() * e.target.scaleX(),
			height: e.target.height() * e.target.scaleY(),
		})
	}, [dispatch])

	let new_obj_tool = useSelector(state => state.new_obj_tool)

	let obj = useSelector(state => {
		if (!state.selected_doc) {
			return null
		}
		return state.docs[state.selected_doc].objs[id]
	})

	if (!obj) {
		return null
	}

	if (obj.type == 'text') {
		return <TextObj obj={obj} draggable={!new_obj_tool} onDragMove={onDragMove} onClick={onClick} onTransform={onTransform} ref={ref}/>
	} else if (obj.type == 'tag') {
		return <TagObj obj={obj} draggable={!new_obj_tool} onDragMove={onDragMove} onClick={onClick} onTransform={onTransform} ref={ref}/>
	} else {
		throw "Invalid obj type"
	}
}

export default React.forwardRef(Obj)