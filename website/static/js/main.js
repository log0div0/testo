
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

setup_toc()
