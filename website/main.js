
import express from 'express'
import assert from 'assert'
import sass from 'node-sass'
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import Home from './home'
import renderMDX from './renderMDX'

const app = express()
const port = 80

app.get('/', (req, res) => {
	res.send('<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(<Home/>))
})

app.get('/main.css', (req, res) => {
	sass.render({file: 'main.scss'}, function(err, result) {
		if (err) {
			return res.status(500).send(err.message)
		}
		res.set('Content-Type', 'text/css')
		res.send(result.css)
	})
})

app.get('/docs/*', async (req, res) => {
	let url = decodeURI(req.originalUrl)
	const result = await renderMDX('docs', `.${url}.md`)
	res.send('<!DOCTYPE html>' + result)
})

app.use(express.static('public'))
app.listen(port, () => console.log(`Listening on port ${port}`))
