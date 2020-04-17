
import React from 'react'
import MDX from '@mdx-js/runtime'
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
			page.url = "/" + path.dirname(categoryRoot) + "/" + result.id + "/" + page.id
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

const getText=(x)=>{
	if (Array.isArray(x)) {
		return x.map(elem => {
			return getText(elem)
		}).join('')
	} else if ((typeof x == "string") || (x instanceof String)) {
		return x
	} else {
		return getText(x.props.children)
	}
}

function H2({children}) {
	let text = getText(children)
	let id = text.replace(/\s+/g, '-').toLowerCase()
	let href = '#' + id
	return (
		<h2>
			<a className="anchor" aria-hidden="true" id={id}></a>
			<a href={href} aria-hidden="true" className="hash-link">
				<svg className="hash-link-icon" aria-hidden="true" height="16" version="1.1" viewBox="0 0 16 16" width="16">
					<path fillRule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path>
				</svg>
			</a>
			{children}
		</h2>
	)
}

function H3({children}) {
	let text = getText(children)
	let id = text.replace(/\s+/g, '-').toLowerCase()
	let href = '#' + id
	return (
		<h3>
			<a className="anchor" aria-hidden="true" id={id}></a>
			<a href={href} aria-hidden="true" className="hash-link">
				<svg className="hash-link-icon" aria-hidden="true" height="16" version="1.1" viewBox="0 0 16 16" width="16">
					<path fillRule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path>
				</svg>
			</a>
			{children}
		</h3>
	)
}

function NavGroup({category}) {
	let navListItems = category.pages.map((page, index) => {
		return (
			<li key={index} className="navListItem">
				<a className="navItem" href={page.url}>{page.name}</a>
			</li>
		)
	})

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
			<ul className="hide">
				{navListItems}
			</ul>
		</div>
	)
}

function DocsLayout({children, toc, prevPage, nextPage}) {
	let navGroups = toc.map((category, index) => <NavGroup key={index} category={category}/>)
	let prevButton = null
	if (prevPage) {
		prevButton = (
			<a className="docs-prev button" href={prevPage.url}>
				<span className="arrow-prev">← </span>
				<span>{prevPage.name}</span>
			</a>
		)
	}
	let nextButtom = null
	if (nextPage) {
		nextButtom = (
			<a className="docs-next button" href={nextPage.url}>
				<span>{nextPage.name}</span>
				<span className="arrow-next"> →</span>
			</a>
		)

	}
	return (
		<Layout>
			<div className="docMainWrapper wrapper">
				<div className="docsNavContainer">
					<nav className="toc">
						<section className="navWrapper wrapper">
							<div className="navGroups">
								{navGroups}
							</div>
						</section>
					</nav>
				</div>
				<div className="mainContainer">
					<div className="wrapper">
						<div className="post">
							{children}
						</div>
						<div className="docs-prevnext">
							{prevButton}
							{nextButtom}
						</div>
					</div>
				</div>
				<nav className="onPageNav">
					<ul className="toc-headings">

					</ul>
				</nav>
			</div>
		</Layout>
	)
}

module.exports = async function(docsRoot, category_id, page_id) {
	const toc = await makeTOC(docsRoot)
	for (let i = 0; i < toc.length; ++i) {
		let category = toc[i];
		if (category.id != category_id) {
			continue
		}
		for (let j = 0; j < category.pages.length; ++j) {
			let page = category.pages[j];
			if (page.id != page_id) {
				continue
			}
			const content = await fs.promises.readFile(page.file_path)
			const components = {
				h1: H1,
				h2: H2,
				h3: H3
			}
			let prevPage = null
			if (j > 0) {
				prevPage = category.pages[j-1]
			} else {
				if (i > 0) {
					let prevCategory = toc[i-1]
					prevPage = prevCategory.pages[prevCategory.pages.length - 1]
				}
			}
			let nextPage = null
			if (j < (category.pages.length - 1)) {
				nextPage = category.pages[j+1]
			} else {
				if (i < (toc.length - 1)) {
					let nextCategory = toc[i+1]
					nextPage = nextCategory.pages[0]
				}
			}
			return (
				<DocsLayout toc={toc} prevPage={prevPage} nextPage={nextPage}>
					<MDX components={components}>{content}</MDX>
				</DocsLayout>
			)
		}
	}
	return <PageNotFound/>
}
