
import React from 'react'
import Drawing from './Drawing'
import Toolbar from './ToolBar'
import ObjPropsBar from './ObjPropsBar'

import {global} from 'styled-jsx/css'

let style = global`
	body {
		padding: 0;
		margin: 0;
		overflow: hidden;
	}

	#app {
		position: relative;
		height: 100%;
	}

	#tool-bar {
		background-color: #454E69;
		height: 40px;
	}

	#drawing {
		background-color: #BBB;
		height: calc(100% - 40px);
	}

	#obj-props-bar {
		position: absolute;
		bottom: 0;
		left: 0;
		height: 60px;
		width: 100%;
		background-color: #F0F0F0;
		box-sizing: border-box;
	}
`

function Doc() {
	return (
		<>
			<style jsx global>{style}</style>
			<Toolbar/>
			<Drawing/>
			<ObjPropsBar/>
		</>
	)
}

export default Doc
