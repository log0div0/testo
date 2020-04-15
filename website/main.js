
import express from 'express'
import assert from 'assert'
import sass from 'node-sass'
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import Home from './js/Home'
import makeDocPage from './js/makeDocPage'

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

app.get('/docs/:category_id/:page_id', async (req, res) => {
	const page = await makeDocPage('docs', req.params.category_id, req.params.page_id)
	res.send('<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(page))
})

app.use(express.static('public'))
app.listen(port, () => console.log(`Listening on port ${port}`))
