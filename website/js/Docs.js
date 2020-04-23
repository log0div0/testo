
import React from 'react'
import MDX from '@mdx-js/runtime'
import fs from 'fs'
import path from 'path'
import * as babel from "@babel/core"
import Layout from './Layout'
import PageNotFound from './PageNotFound'
import hljs from 'highlight.js'

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

function Pre(props) {
	return (
		<pre {...props} />
	)
}

function TestoDef(hljs) {
	return {
		name: 'Testo Lang',
		keywords: 'for if else continue break machine',
			contains: [
				{
					className: 'string',
					variants: [
					 	{
							begin: /"""/,
							end: /"""/
					 	},
					 	{
							begin: /"/,
							end: /"/
					 	}
					],
					contains: [
						{
							className: 'literal',
							variants: [
								{ begin: '\\\\"' },
								{ begin: '\\\\n' }
							]
						},
						{
							className: 'subst',
							begin: "\\${",
							end: "}"
						}
					]
				},
				hljs.C_BLOCK_COMMENT_MODE,
				hljs.HASH_COMMENT_MODE,
				{
					className: 'number',
					begin: /\b\d+(Kb|Mb|Gb|s|m|h)?/
				},
				{
					className: 'function',
					beginKeywords: 'machine test network flash',
					end: /{/,
					excludeEnd: true,
					contains: [
						hljs.UNDERSCORE_TITLE_MODE
					]
				},
				{
					className: 'function',
					keywords: 'param',
					begin: "param\\s+",
					contains: [
						hljs.UNDERSCORE_TITLE_MODE
					]
				},
				{
					className: 'attribute',
					begin: /type|wait|press|plug|unplug|start|stop|exec|copyto|copyfrom|shutdown|print|abort|mouse|sleep/
				}
			]
	}
}

hljs.registerLanguage('testo', TestoDef)

function Code(props) {
	let lang = 'text'
	if (props.className) {
		lang = props.className.substr("language-".length)
	}
	let {value} = hljs.highlight(lang, props.children)
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
				h1: H1,
				h2: H2,
				h3: H3,
				Terminal,
				pre: Pre,
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
