
import React from 'react'
import MDX from '@mdx-js/runtime'
import fs from 'fs'
import Layout from './components/Layout'
import PageNotFound from './PageNotFound'
import HighlightTesto from './components/HighlightTesto'
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

function makeHeader(Tag) {
	return function({children}) {
		let text = getText(children)
		let id = text.replace(/\s+/g, '-').toLowerCase()
		let href = '#' + id
		return (
			<Tag>
				<a id={id} className="anchor"/>
				<a href={href} className="anchor-icon"/>
				{children}
			</Tag>
		)
	}
}

function Book({book, currentChapterUrl}) {
	let className = "book close"
	let chapters = book.chapters.map((chapter, index) => {
		if (chapter.url == currentChapterUrl) {
			className = "book"
		}
		return <a key={index} href={chapter.url}>{chapter.name}</a>
	})

	return (
		<div className={className}>
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

function DocsLayout({children, toc, prevChapter, nextChapter, currentChapterUrl}) {
	let books = toc.books.map((book, index) => {
		return <Book key={index} book={book} currentChapterUrl={currentChapterUrl}/>
	})
	let prevButton = null
	if (prevChapter) {
		prevButton = <a id="docs-prev-button" href={prevChapter.url}>← {prevChapter.name}</a>
	}
	let nextButtom = null
	if (nextChapter) {
		nextButtom = <a id="docs-next-button" href={nextChapter.url}>{nextChapter.name} →</a>
	}
	return (
		<Layout>
			<div id="docs">
				<section id="docs-toc">
					<div className="container">
						{books}
					</div>
				</section>
				<section id="docs-article">
					{children}
					<div>
						{prevButton}
						{nextButtom}
					</div>
				</section>
				<section id="docs-minitoc">
				</section>
			</div>
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
		<div className="terminal">
			<header>
				<div className="red button"></div>
				<div className="yellow button"></div>
				<div className="green button"></div>
			</header>
			<main style={{height: height}}>
				{children}
			</main>
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
				h2: makeHeader("h2"),
				h3: makeHeader("h3"),
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
				<DocsLayout toc={toc} prevChapter={prevChapter} nextChapter={nextChapter} currentChapterUrl={chapter_url}>
					<MDX components={components}>{content}</MDX>
				</DocsLayout>
			)
		}
	}
	return <PageNotFound/>
}
