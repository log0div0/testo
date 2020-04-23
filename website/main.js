
import express from 'express'
import assert from 'assert'
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import Home from './js/Home'
import {makeDocPage, makeDocToc} from './js/Docs'

const app = express()
const port = 3000

app.get('/', (req, res) => {
	res.send('<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(<Home/>))
})

app.get('/docs/*', async (req, res) => {
	const page = await makeDocPage(docsToc, req.originalUrl)
	res.send('<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(page))
})

app.get('/tutorials/*', async (req, res) => {
	const page = await makeDocPage(tutorialsToc, req.originalUrl)
	res.send('<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(page))
})

app.use('/static', express.static('static'))

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
		"/tutorials/7_ping"
	]
}

async function main() {
	docsToc = await makeDocToc(docsToc)
	tutorialsToc = await makeDocToc(tutorialsToc)

	app.listen(port, () => console.log(`Listening on port ${port}`))
}

main()
