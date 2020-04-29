
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
			// a.href = element.querySelector('.hash-link').href
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

setup_toc()
setup_minitoc()