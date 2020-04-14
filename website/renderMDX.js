
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import mdx from '@mdx-js/mdx'
import {MDXProvider, mdx as createElement} from '@mdx-js/react'
import fs from 'fs'
import * as babel from "@babel/core"
import DocsLayout from './components/DocsLayout'

function H1({children}) {
	return <h1 style={{color: 'green'}}>{children}</h1>
}

module.exports = async function(docFile) {
	const content = await fs.promises.readFile(docFile);
	const jsx = await mdx(content, { skipExport: true });
	const {code} = babel.transform(jsx, {
		presets: [
			"@babel/preset-env",
			"@babel/preset-react"
		]
	});

	const scope = {mdx: createElement, require}

	const fn = new Function(
		'React',
		...Object.keys(scope),
		`${code}; return React.createElement(MDXContent)`
	)
	const element = fn(React, ...Object.values(scope))
	const components = {
		h1: H1
	}
	const elementWithProvider = React.createElement(
		MDXProvider,
		{components},
		element
	)
	const page = (
		<DocsLayout>
			{elementWithProvider}
		</DocsLayout>
	)
	return ReactDOMServer.renderToStaticMarkup(page)
}
