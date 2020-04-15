
import React from 'react'
import mdx from '@mdx-js/mdx'
import {MDXProvider, mdx as createElement} from '@mdx-js/react'
import fs from 'fs'
import path from 'path'
import * as babel from "@babel/core"
import Layout from './Layout'
import PageNotFound from './PageNotFound'

async function makePage(file_path) {
	let parsed = path.parse(file_path)
	if (parsed.ext != ".md") {
		return null
	}
	let indexOfDash = parsed.name.indexOf('-')
	if (indexOfDash < 0) {
		return null
	}
	return {
		id: parsed.name.substring(0, indexOfDash).trim(),
		name: parsed.name.substring(indexOfDash + 1, parsed.name.length).trim(),
		file_path
	}
}

async function makeCategory(categoryRoot) {
	const stats = await fs.promises.stat(categoryRoot)
	if (!stats.isDirectory()) {
		return null
	}
	let basename = path.basename(categoryRoot)
	let indexOfDash = basename.indexOf('-')
	if (indexOfDash < 0) {
		return null
	}
	let result = {
		id: basename.substring(0, indexOfDash).trim(),
		name: basename.substring(indexOfDash + 1, basename.length).trim(),
		pages: []
	}
	const files = await fs.promises.readdir(categoryRoot)
	for (let file of files) {
		let page = await makePage(path.join(categoryRoot, file))
		if (page) {
			result.pages.push(page)
		}
	}
	return result
}

async function makeTOC(docsRoot) {
	let result = []
	const files = await fs.promises.readdir(docsRoot)
	for (let file of files) {
		let category = await makeCategory(path.join(docsRoot, file))
		if (category) {
			result.push(category)
		}
	}
	return result
}

function H1({children}) {
	return <h1 className="postHeaderTitle">{children}</h1>
}

async function MDXtoReact(mdxFile) {
	const content = await fs.promises.readFile(mdxFile)
	const jsx = await mdx(content, { skipExport: true })
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
	return React.createElement(MDXProvider, {components}, element)
}

function NavListItem({page}) {
	return (
		<li className="navListItem navListItemActive">
			<a className="navItem" href="/docs/en/installation">{page.name}</a>
		</li>
	)
}

function NavGroup({category}) {
	let navListItems = category.pages.map((page, index) => <NavListItem key={index} page={page}/>)
	return (
		<div className="navGroup">
			<h3 className="navGroupCategoryTitle collapsible">
				{category.name}
				<span className="arrow">
					<svg width="24" height="24" viewBox="0 0 24 24">
						<path fill="#565656" d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"></path>
						<path d="M0 0h24v24H0z" fill="none"></path>
					</svg>
				</span>
			</h3>
			<ul className="">
				{navListItems}
			</ul>
		</div>
	)
}

function DocsLayout({children, toc}) {
	let navGroups = toc.map((category, index) => <NavGroup key={index} category={category}/>)
	return (
		<Layout>
			<div className="docMainWrapper wrapper">
				<div className="docsNavContainer">
					<nav className="toc">
						<div className="toggleNav">
							<section className="navWrapper wrapper">
								<div className="navGroups">
									{navGroups}
								</div>
							</section>
						</div>
					</nav>
				</div>
				<div className="container mainContainer docsContainer">
					<div className="wrapper">
						<div className="post">
							{children}
						</div>
					</div>
				</div>
				<nav className="onPageNav"></nav>
			</div>
		</Layout>
	)
}

module.exports = async function(docsRoot, category_id, page_id) {
	const toc = await makeTOC(docsRoot)
	for (let category of toc) {
		if (category.id != category_id) {
			continue
		}
		for (let page of category.pages) {
			if (page.id != page_id) {
				continue
			}
			const doc = await MDXtoReact(page.file_path)
			return <DocsLayout toc={toc}>{doc}</DocsLayout>
		}
	}
	return <PageNotFound/>
}
