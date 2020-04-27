
import React from 'react'
import MDX from '@mdx-js/runtime'
import fs from 'fs'
import Layout from './Layout'
import PageNotFound from './PageNotFound'
import HighlightTesto from './HighlightTesto'
import hljs from 'highlight.js'

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
	return <h2>{children}</h2>
}

function H3({children}) {
	let text = getText(children)
	let id = text.replace(/\s+/g, '-').toLowerCase()
	let href = '#' + id
	return <h3>{children}</h3>
}

function NavGroup({category}) {
	let navGroupItems = category.pages.map((page, index) => {
		return (
			<li key={index}>
				<a href={page.url}>{page.name}</a>
			</li>
		)
	})

	return (
		<div>
			<h1>
				{category.name}
			</h1>
			<ul>
				{navGroupItems}
			</ul>
		</div>
	)
}

function DocsLayout({children, toc, prevPage, nextPage}) {
	let navGroups = toc.map((category, index) => <NavGroup key={index} category={category}/>)
	let prevButton = null
	if (prevPage) {
		prevButton = <a href={prevPage.url}>← {prevPage.name}</a>
	}
	let nextButtom = null
	if (nextPage) {
		nextButtom = <a href={nextPage.url}>{nextPage.name} →</a>
	}
	return (
		<Layout>
			<main>
				<section className="docs-left">
					<nav>
						{navGroups}
					</nav>
				</section>
				<section className="docs-center">
					<main className="docs-article">
						{children}
					</main>
					<footer>
						{prevButton}
						{nextButtom}
					</footer>
				</section>
				<section className="docs-right">
					<nav>
					</nav>
				</section>
			</main>
		</Layout>
	)
}

export async function makeDocToc(src) {
	let result = []
	for (let category_name in src) {
		let category = {
			name: category_name,
			pages: []
		}
		for (let page_url of src[category_name]) {
			let page = {
				url: page_url,
				file_path: `.${page_url}.md`
			}
			const content = await fs.promises.readFile(page.file_path, 'utf-8')
			let first_line = content.split(/\r?\n/)[0]
			if (!first_line.startsWith("# ")) {
				throw `The first line of file ${page.file_path} must starts with #`
			}
			page.name = first_line.substr(1).trim()
			category.pages.push(page)
		}
		result.push(category)
	}
	return result
}

function Terminal({children, height}) {
	height = height ? height : "300px"
	return (
		<div className="terminal-window" style={{height: height}}>
			<header>
				<div className="terminal-button red"></div>
				<div className="terminal-button yellow"></div>
				<div className="terminal-button green"></div>
			</header>
			<section className="terminal">
				{children}
			</section>
		</div>
	)
}

hljs.registerLanguage('testo', HighlightTesto)

function Code({children, className}) {
	if (!className) {
		return <code>{children}</code>
	}
	let lang = className.substr("language-".length)
	let {value} = hljs.highlight(lang, getText(children))
	return (
		<code className="hljs" dangerouslySetInnerHTML={{__html: value}}/>
	)
}

export async function makeDocPage(toc, page_url) {
	for (let i = 0; i < toc.length; ++i) {
		let category = toc[i];
		for (let j = 0; j < category.pages.length; ++j) {
			let page = category.pages[j];
			if (page.url != page_url) {
				continue
			}
			const content = await fs.promises.readFile(page.file_path)
			const components = {
				h2: H2,
				h3: H3,
				Terminal,
				code: Code
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
