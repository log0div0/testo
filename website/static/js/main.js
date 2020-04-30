
function setup_header() {
	if (location.pathname.startsWith("/docs")) {
		document.querySelector('body > header a[href^="/docs"]').className = "active";
	}

	if (location.pathname.startsWith("/tutorials")) {
		document.querySelector('body > header a[href^="/tutorials"]').className = "active";
	}
}

function setup_toc() {
	let toc = document.getElementById(`docs-toc`)
	if (!toc) {
		return
	}
	let books = document.querySelectorAll('.book')
	for (let book of books) {
		let name = book.querySelector('.name')
		let chapters = book.querySelector('.chapters')
		let toggle_book = function() {
			book.classList.toggle('close')
		}
		name.addEventListener('click', toggle_book)
		let a = chapters.querySelector(`a[href="${location.pathname}"]`)
		if (a) {
			a.classList.toggle('active')
		}
	}
}

function setup_minitoc() {
	let article = document.getElementById("docs-article")
	let container = document.createElement("div")
	container.className = 'container'
	for (let element of article.children) {
		if ((element.tagName == "H2") || (element.tagName == "H3")) {
			let a = document.createElement('a')
			a.href = element.querySelector('.anchor-icon').href
			a.innerText = element.innerText
			a.className = element.tagName
			container.appendChild(a)
		}
	}
	if (!container.children.length) {
		return
	}
	let minitoc = document.getElementById("docs-minitoc")
	minitoc.appendChild(container)
}

function setup_minitoc_scroll_spy() {
	let sections = document.querySelectorAll("#docs-minitoc a")
	if (!sections.length) {
		return
	}
	let getActiveSectionIndex = function() {
		for (let i = 0; i < sections.length; i++) {
			let id = decodeURIComponent(sections[i].href.split('#')[1])
			let anchor = document.getElementById(id)
			let {top} = anchor.getBoundingClientRect()
			if (top > 0) {
				if (i > 0) {
					return i - 1
				} else {
					return i
				}
			}
		}
		return sections.length - 1
	}
	let timer
	let onScroll = function() {
		if (timer) {
			return
		}
		timer = setTimeout(function() {
			timer = null
			for (let i = 0; i < sections.length; i++) {
				sections[i].classList.remove('active')
			}
			let index = getActiveSectionIndex()
			sections[index].classList.add('active')
		}, 100)
	}

	document.addEventListener('scroll', onScroll)
	document.addEventListener('resize', onScroll)
	onScroll()
}

setup_header()
setup_toc()
setup_minitoc()
setup_minitoc_scroll_spy()