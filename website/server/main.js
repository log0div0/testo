
import express from 'express'
import assert from 'assert'
import sass from 'node-sass'
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import Home from './Home'
import {makeDocChapter, makeDocToc} from './Docs'
import PageNotFound from './PageNotFound'
import Downloads from './Downloads'
import Features from './Features'
import Buy from './Buy'
import Examples from './Examples'
import Contact from './Contact'

const app = express()
const port = 3000

function renderHtml(page) {
	return '<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(page)
}

app.get('/', (req, res) => {
	res.send(renderHtml(<Home/>))
})

app.get('/downloads', (req, res) => {
	res.send(renderHtml(<Downloads/>))
})

app.get('/buy', (req, res) => {
	res.send(renderHtml(<Buy/>))
})

app.get('/features', (req, res) => {
	res.send(renderHtml(<Features/>))
})

app.get('/examples', (req, res) => {
	res.send(renderHtml(<Examples/>))
})

app.get('/contact', (req, res) => {
	res.send(renderHtml(<Contact/>))
})

app.get('/main.css', (req, res) => {
	sass.render({file: './styles/main.scss'}, function(err, result) {
		if (err) {
			console.log(err)
			return res.status(500).send(err.message)
		}
		res.set('Content-Type', 'text/css')
		res.send(result.css)
	})
})

app.get('/docs/*', async (req, res) => {
	const page = await makeDocChapter(docsToc, req.originalUrl)
	res.send(renderHtml(page))
})

app.get('/tutorials/*', async (req, res) => {
	const page = await makeDocChapter(tutorialsToc, req.originalUrl)
	res.send(renderHtml(page))
})

app.use('/static', express.static('static'))

app.use(function(req, res, next) {
	res.status(404)
	const page = <PageNotFound/>
	res.send(renderHtml(page));
});

let docsToc = {
	"Общие сведения": [
		"/docs/getting_started/intro",
		"/docs/getting_started/getting_started",
		"/docs/getting_started/architecture",
		"/docs/getting_started/starting",
		"/docs/getting_started/test_policy"
	],
	"Тестовые сценарии": [
		"/docs/lang/general",
		"/docs/lang/lexems",
		"/docs/lang/var_refs",
		"/docs/lang/machine",
		"/docs/lang/flash",
		"/docs/lang/network",
		"/docs/lang/param",
		"/docs/lang/test",
		"/docs/lang/macro",
		"/docs/lang/actions",
		"/docs/lang/mouse",
		"/docs/lang/if",
		"/docs/lang/for",
		"/docs/lang/keys"
	],
	"Запросы на языке Javascript": [
		"/docs/js/general",
		"/docs/js/global_funcs",
		"/docs/js/tensor_textline",
		"/docs/js/point"
	],
	"История изменений": [
		"/docs/rn/1.5.0",
		"/docs/rn/1.4.0"
	]
}

let tutorialsToc = {
	"Обучающие материалы": [
		"/tutorials/1_creating_vm",
		"/tutorials/2_ubuntu_installation",
		"/tutorials/3_guest_additions",
		"/tutorials/4_params",
		"/tutorials/5_caching",
		"/tutorials/6_nat",
		"/tutorials/7_ping",
		"/tutorials/8_flash",
		"/tutorials/9_macros",
		"/tutorials/10_if",
	]
}

async function main() {
	docsToc = await makeDocToc(docsToc)
	tutorialsToc = await makeDocToc(tutorialsToc)

	app.listen(port, () => console.log(`Listening on port ${port}`))
}

main()
