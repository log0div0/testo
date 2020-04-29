
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
			toggle_book()
		}
	}
}

function setup_minitoc() {
	let article = document.getElementById("docs-article")
	let div = document.createElement("div")
	let sections = []
	for (let element of article.children) {
		if ((element.tagName == "H2") || (element.tagName == "H3")) {
			let a = document.createElement('a')
			// a.href = element.querySelector('.hash-link').href
			a.innerText = element.innerText
			a.className = element.tagName
			div.appendChild(a)
		}
	}
	let minitoc = document.getElementById("docs-minitoc")
	minitoc.appendChild(div)
}

setup_toc()
setup_minitoc()