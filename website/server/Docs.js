
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

function Book({book}) {
	let chapters = book.chapters.map((chapter, index) => {
		return <a key={index} href={chapter.url}>{chapter.name}</a>
	})

	return (
		<div className="book close">
			<div className="name">
				<h1>{book.name}</h1>
				<div className="arrow"/>
			</div>
			<div className="chapters">
				{chapters}
			</div>
		</div>
	)
}

function DocsLayout({children, toc, prevChapter, nextChapter}) {
	let books = toc.books.map((book, index) => {
		return <Book key={index} book={book}/>
	})
	let prevButton = null
	if (prevChapter) {
		prevButton = <a href={prevChapter.url}>← {prevChapter.name}</a>
	}
	let nextButtom = null
	if (nextChapter) {
		nextButtom = <a href={nextChapter.url}>{nextChapter.name} →</a>
	}
	return (
		<Layout>
			<main>
				<section id="docs-toc">
					{books}
				</section>
				<section id="docs-article">
					{children}
					<div id="prev-next-buttons">
						{prevButton}
						{nextButtom}
					</div>
				</section>
				<section id="docs-minitoc">
				</section>
			</main>
		</Layout>
	)
}

export async function makeDocToc(src) {
	let toc = {
		books: []
	}
	for (let book_name in src) {
		let book = {
			name: book_name,
			chapters: []
		}
		for (let chapter_url of src[book_name]) {
			let chapter = {
				url: chapter_url,
				file_path: `.${chapter_url}.md`
			}
			const content = await fs.promises.readFile(chapter.file_path, 'utf-8')
			let first_line = content.split(/\r?\n/)[0]
			if (!first_line.startsWith("# ")) {
				throw `The first line of file ${chapter.file_path} must starts with #`
			}
			chapter.name = first_line.substr(1).trim()
			book.chapters.push(chapter)
		}
		toc.books.push(book)
	}
	return toc
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

export async function renderDocChapter(toc, chapter_url) {
	for (let i = 0; i < toc.books.length; ++i) {
		let book = toc.books[i];
		for (let j = 0; j < book.chapters.length; ++j) {
			let chapter = book.chapters[j];
			if (chapter.url != chapter_url) {
				continue
			}
			const content = await fs.promises.readFile(chapter.file_path)
			const components = {
				h2: H2,
				h3: H3,
				Terminal,
				code: Code
			}
			let prevChapter = null
			if (j > 0) {
				prevChapter = book.chapters[j-1]
			} else {
				if (i > 0) {
					let prevBook = toc.books[i-1]
					prevChapter = prevBook.chapters[prevBook.chapters.length - 1]
				}
			}
			let nextChapter = null
			if (j < (book.chapters.length - 1)) {
				nextChapter = book.chapters[j+1]
			} else {
				if (i < (toc.books.length - 1)) {
					let nextBook = toc.books[i+1]
					nextChapter = nextBook.chapters[0]
				}
			}
			return (
				<DocsLayout toc={toc} prevChapter={prevChapter} nextChapter={nextChapter}>
					<MDX components={components}>{content}</MDX>
				</DocsLayout>
			)
		}
	}
	return <PageNotFound/>
}
