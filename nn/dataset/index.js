
import fs from 'fs'
import path from 'path'
import puppeteer from 'puppeteer'
import React from 'react'
import ReactDOMServer from 'react-dom/server'
import * as fonts from './fonts'
import {randomArrayElement} from './common'

function App(props) {

	let fontFamily = randomArrayElement(fonts.families)
	// let fontFamily = 'Segoe UI'

	let bodyStyle = {
		margin: 0,
		padding: 0,
		fontFamily: fontFamily
	}

	return (
		<html>
			<head>
				<meta charSet="utf-8"/>
				<style dangerouslySetInnerHTML={{__html: fonts.css}}></style>
			</head>
			<body style={bodyStyle}>
				{props.children}
			</body>
		</html>
	)
}

function generate_label() {
	let background = document.querySelector('#background')
	let {x: min_x, y: min_y, width: max_x, height: max_y} = background.getBoundingClientRect()
	if ((min_x != 0) || (min_y != 0)) {
		throw Error("Invalid background position")
	}
	let label = {
		textlines: []
	}
	let textline_nodes = document.querySelectorAll(".textline")
	for (let textline_node of textline_nodes) {
		let {x, y, top, bottom, left, right, width, height} = textline_node.getBoundingClientRect()
		let textline = {
			text: '',
			bbox: {x, y, top, bottom, left, right, width, height},
			chars: []
		}
		let char_nodes = textline_node.querySelectorAll('.char')
		for (let char_node of char_nodes) {
			let text = char_node.innerText;
			if (text == '') {
				text = ' '
			}
			let {x, y, top, bottom, left, right, width, height} = char_node.getBoundingClientRect()
			let char = {
				text: text,
				bbox: {x, y, top, bottom, left, right, width, height}
			}
			if ((top < min_y) || (bottom > max_y)) {
				throw Error("(top < min_y) || (bottom > max_y)")
			}
			if ((left < min_x) || (right > max_x)) {
				continue
			}
			if ((width == 0) || (height == 0)) {
				throw Error("(width == 0) || (height == 0)")
			}
			textline.text += text
			textline.chars.push(char)
		}
		label.textlines.push(textline)
	}
	return label
}

const examples_per_template = 200

async function generate_dataset(src_path) {
	let src = path.parse(src_path)
	let dst_dir = path.join(src.dir.replace("input/", "output/"), src.name)
	await fs.promises.mkdir(dst_dir, {recursive: true})

	let browser = await puppeteer.launch()
	let page = await browser.newPage()
	try {
		let {Example} = require('./' + src_path)
		for (let i = 0; i < examples_per_template; i++) {
			process.stdout.write("\r" + src_path + `: ${i+1}/${examples_per_template}`)
			let errors_count = 0
			while (true) {
				try {
					let html = '<!DOCTYPE html>' + ReactDOMServer.renderToStaticMarkup(<App><Example/></App>)
					let html_path = path.join(dst_dir, `${i}.html`)
					await fs.promises.writeFile(html_path, html)
					await page.goto('file://' + path.resolve(html_path))
					let clip = await page.evaluate(() => {
						let background = document.querySelector('#background')
						let {x, y, width, height} = background.getBoundingClientRect()
						return {x, y, width, height}
					})
					await page.setViewport({width: clip.width, height: clip.height})
					let screenshot = await page.screenshot({clip})
					await fs.promises.writeFile(path.join(dst_dir, `${i}.png`), screenshot, {encoding: 'binary'})
					let label = await page.evaluate(generate_label)
					await fs.promises.writeFile(path.join(dst_dir, `${i}.json`), JSON.stringify(label, null, 2))
				} catch (error) {
					if (!error.message.includes("(width == 0) || (height == 0)")) {
						throw error
					}
					errors_count += 1
					if (errors_count > 10) {
						throw error
					}
					continue
				}
				break
			}
		}
		process.stdout.write('\n')
	} finally {
		await page.close()
		await browser.close()
	}
}

async function walk(dir) {
	let files = await fs.promises.readdir(dir)
	for (let file of files) {
		if (file[0] == '_') {
			continue
		}
		let file_path = path.join(dir, file)
		let stat = await fs.promises.stat(file_path)
		if (stat.isDirectory()) {
			await walk(file_path)
		} else {
			if (path.extname(file_path) == '.js') {
				await generate_dataset(file_path)
			}
		}
	}
}

(async function() {
	try {
		await fs.promises.rmdir('output', {recursive: true})
		await walk('input')
	} catch (e) {
		console.log(e)
		process.exit(-1)
	}
	console.log('OK')
	process.exit(0)
})()
