
import React, {useEffect, useState, useRef, useCallback} from 'react'
import { Stage } from 'react-konva'
import { ReactReduxContext, Provider, useSelector, useDispatch } from 'react-redux'
import LayerBackground from './LayerBackground'
import LayerForeground from './LayerForeground'
import useImage from 'use-image'
import path from 'path'

function useElementSize(ref) {
	let [size, setSize] = useState({})
	useEffect(() => {
		function handleResize() {
			setSize({
				width: ref.current.getBoundingClientRect().width,
				height: ref.current.getBoundingClientRect().height,
			})
		}
		handleResize()
		window.addEventListener("resize", handleResize)
		return () => window.removeEventListener("resize", handleResize)
	}, [ref])
	return size
}

function getRelativePointerPosition(node) {
	var transform = node.getAbsoluteTransform().copy();
	// to detect relative position we need to invert transform
	transform.invert();

	// get pointer (say mouse or touch) position
	var pos = node.getStage().getPointerPosition();

	// now we can find relative point
	return transform.point(pos);
}

function Drawing() {
	let drawingRef = useRef(null)
	let drawingSize = useElementSize(drawingRef)

	let dispatch = useDispatch()

	useEffect(() => {
		function eventHandler(e) {
			dispatch({
				type: 'KEYBOARD_EVENT',
				isDown: e.type == "keydown",
				code: e.code,
				ctrlKey: e.ctrlKey
			})
		}
		document.addEventListener('keydown', eventHandler)
		document.addEventListener('keyup', eventHandler)
		return () => {
			document.removeEventListener('keydown', eventHandler)
			document.removeEventListener('keyup', eventHandler)
		}
	}, [])

	let onWheel = useCallback(e => {
		let stage = e.currentTarget

		var oldScale = stage.scaleX()

		var pointer = stage.getPointerPosition()

		var mousePointTo = {
			x: (pointer.x - stage.x()) / oldScale,
			y: (pointer.y - stage.y()) / oldScale,
		}

		let scaleBy = 1.1

		var newScale =
			e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy

		stage.scale({ x: newScale, y: newScale })

		var newPos = {
			x: pointer.x - mousePointTo.x * newScale,
			y: pointer.y - mousePointTo.y * newScale,
		}
		stage.position(newPos)
		stage.batchDraw()

		dispatch({
			type: "NEW_SCALE",
			value: newScale
		})
	})

	const selected_doc = useSelector(state => state.selected_doc)
	let img_url = selected_doc ? path.join(DATASET_DIR, selected_doc + '.png') : null
	const [image] = useImage(img_url)

	let stageRef = useRef(null)

	useEffect(() => {
		if (!image) {
			return
		}
		if (!stageRef.current) {
			return
		}
		let scaleX = drawingSize.width / image.width
		let scaleY = drawingSize.height / image.height
		let scale = Math.min(scaleX, scaleY)
		let im_width = image.width
		let im_height = image.height
		if (scale < 1) {
			im_width *= scale
			im_height *= scale
			stageRef.current.scale({x: scale, y: scale})
			dispatch({
				type: "NEW_SCALE",
				value: scale
			})
		} else {
			scale = 1
			stageRef.current.scale({x: scale, y: scale})
			dispatch({
				type: "NEW_SCALE",
				value: scale
			})
		}
		stageRef.current.setX((drawingSize.width - im_width) / 2)
		stageRef.current.setY((drawingSize.height - im_height) / 2)
		stageRef.current.batchDraw()
	}, [image])

	let onMouse = useCallback(e => {
		if ((e.evt.buttons == 2) && (e.evt.type == "mousemove")) {
			stageRef.current.setX(stageRef.current.x() + e.evt.movementX)
			stageRef.current.setY(stageRef.current.y() + e.evt.movementY)
			stageRef.current.batchDraw()
			return
		}
		if ((e.evt.buttons == 0) && (e.evt.type == "mousemove")) {
			return
		}
		if ((e.evt.button == 2) && (e.evt.type == "mousedown")) {
			dispatch({
				type: 'SET_SHOW_META',
				value: true
			})
			return
		}
		if ((e.evt.button == 2) && (e.evt.type == "mouseup")) {
			dispatch({
				type: 'SET_SHOW_META',
				value: false
			})
			return
		}
		if (e.evt.button != 0) {
			return
		}
		let pos = getRelativePointerPosition(stageRef.current)
		dispatch({
			type: 'MOUSE_EVENT',
			x: pos.x,
			y: pos.y,
			code: e.type,
			target_name: e.target.name()
		})
	}, [dispatch])

	let onContextMenu = useCallback(e => {
		e.evt.preventDefault()
		e.evt.stopPropagation()
	})

	return (
		<div id="drawing" ref={drawingRef}>
			<ReactReduxContext.Consumer>
				{({ store }) => (
					<Stage
						name="stage"
						ref={stageRef}
						width={drawingSize.width}
						height={drawingSize.height}
						onWheel={onWheel}
						onMouseDown={onMouse}
						onMouseMove={onMouse}
						onMouseUp={onMouse}
						onContextMenu={onContextMenu}
					>
						<Provider store={store}>
							<LayerBackground image={image}/>
							<LayerForeground/>
						</Provider>
					</Stage>
				)}
			</ReactReduxContext.Consumer>
		</div>
	)
}

export default Drawing
