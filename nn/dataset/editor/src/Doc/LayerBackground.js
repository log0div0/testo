
import React, {useCallback} from 'react'
import { Image, Layer } from 'react-konva'
import { connect, useSelector } from 'react-redux'

function LayerBackground({image, dispatch}) {
	const scale = useSelector(state => state.scale)

	if (!image) {
		return null
	}

	return (
		<Layer imageSmoothingEnabled={scale < 1}>
			<Image name="background" image={image}/>
		</Layer>
	)
}

export default connect()(LayerBackground)