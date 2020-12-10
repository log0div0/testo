
import React, { useRef, useEffect, useMemo, useCallback } from 'react'
import { Layer, Transformer } from 'react-konva'
import { useSelector } from 'react-redux'
import Obj from './Obj'

let Selection = React.forwardRef(({}, ref) => {
	let scale = useSelector(state => state.scale)

	let boundBoxFunc = useCallback((oldBoundBox, newBoundBox) => {
		if (newBoundBox.width < 0) {
			newBoundBox.x += newBoundBox.width
			newBoundBox.width = -newBoundBox.width
		}
		if (newBoundBox.height < 0) {
			newBoundBox.y += newBoundBox.height
			newBoundBox.height = -newBoundBox.height
		}
		return newBoundBox
	})

	return (
		<Transformer
			ref={ref}
			keepRatio={false}
			rotateEnabled={false}
			boundBoxFunc={boundBoxFunc}
			ignoreStroke={true}
			anchorSize={Math.min(16, 8 * scale)}
		/>
	)
})

function LayerForeground() {
	let objs_ids = useSelector(state => state.objs_ids)
	let selected_objs = useSelector(state => state.selected_objs)

	let trRef = useRef()

	let selectedRects = []
	let collectRects = (rect) => {
		if (rect) {
			selectedRects.push(rect)
		}
	}

	useEffect(() => {
		if (!trRef.current) {
			return
		}
		trRef.current.nodes(selectedRects)
		trRef.current.getLayer().batchDraw()
	})

	return (
		<Layer>
			{
				objs_ids.map((id) => {
					return <Obj
						ref={selected_objs.includes(id) ? collectRects: null}
						id={id}
						key={id}
					/>
				})
			}
			{
				(selected_objs.length != 0) && (<Selection ref={trRef}/>)
			}
		</Layer>
	)
}

export default LayerForeground