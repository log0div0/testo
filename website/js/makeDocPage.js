
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

function NavGroup({category, category_id, page_id}) {
	let navListItems = category.pages.map((page, index) => {
		let li_class = "navListItem"
		if ((category.id == category_id) && (page.id == page_id)) {
			li_class = "navListItem navListItemActive";
		}
		return (
			<li key={index} className={li_class}>
				<a className="navItem" href={`/docs/${category.id}/${page.id}`}>{page.name}</a>
			</li>
		)
	})
	let arrow_class = "arrow"
	let ul_class = "hide"
	if (category.id == category_id) {
		arrow_class = "arrow rotate"
		ul_class = ""
	}

	return (
		<div className="navGroup">
			<h3 className="navGroupCategoryTitle collapsible">
				{category.name}
				<span className={arrow_class}>
					<svg width="24" height="24" viewBox="0 0 24 24">
						<path fill="#565656" d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"></path>
						<path d="M0 0h24v24H0z" fill="none"></path>
					</svg>
				</span>
			</h3>
			<ul className={ul_class}>
				{navListItems}
			</ul>
		</div>
	)
}

function DocsLayout({children, toc, ...rest}) {
	let navGroups = toc.map((category, index) => <NavGroup key={index} category={category} {...rest}/>)
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
			const content = await fs.promises.readFile(page.file_path)
			const components = {
				h1: H1
			}
			return (
				<DocsLayout toc={toc} category_id={category_id} page_id={page_id}>
					<MDX components={components}>{content}</MDX>
				</DocsLayout>
			)
		}
	}
	return <PageNotFound/>
}
