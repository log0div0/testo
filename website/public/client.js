
var a = document.querySelector(`.navGroups a[href="${location.pathname}"]`)
if (a) {
	var li = a.parentNode;
	li.classList.toggle("navListItemActive")
	var ul = li.parentNode;
	var coll = ul.previousElementSibling
	var arrow = coll.childNodes[1];
	arrow.classList.toggle('rotate');
	var content = coll.nextElementSibling;
	content.classList.toggle('hide');
}

var coll = document.getElementsByClassName('collapsible');
for (var i = 0; i < coll.length; i++) {
	coll[i].addEventListener('click', function() {
		var arrow = this.childNodes[1];
		arrow.classList.toggle('rotate');
		var content = this.nextElementSibling;
		content.classList.toggle('hide');
	});
}

var xxx = document.querySelectorAll(".post > *")
var toc = []
for (let x of xxx) {
	if (x.tagName == "H2") {
		toc.push({
			name: x.innerText,
			href: x.querySelector('.hash-link').href,
			subtoc: []
		})
	}
	if (x.tagName == "H3") {
		if (toc.length == 0) {
			throw "Invalid TOC";
		}
		toc[toc.length - 1].subtoc.push({
			name: x.innerText,
			href: x.querySelector('.hash-link').href,
		})
	}
}

var ul = document.querySelector("ul.toc-headings")
for (var h2 of toc) {
	var li = document.createElement("li")
	var a = `<a href="${h2.href}">${h2.name}</a>`
	var subtoc = ''
	if (h2.subtoc.length) {
		subtoc = '<ul class="toc-headings">'
		for (var h3 of h2.subtoc) {
			subtoc += `<li><a href="${h3.href}">${h3.name}</a></li>`
		}
		subtoc += '</ul>'
	}
	li.innerHTML = a + subtoc
	ul.appendChild(li)
}

(function scrollSpy() {
	var OFFSET = 10;
	var timer;
	var headingsCache;

	var findHeadings = function findHeadings() {
		return headingsCache || document.querySelectorAll('.toc-headings > li > a');
	};

	var onScroll = function onScroll() {
		if (timer) {
			// throttle
			return;
		}

		timer = setTimeout(function () {
			timer = null;
			var activeNavFound = false;
			var headings = findHeadings(); // toc nav anchors

			/**
			 * On every call, try to find header right after  <-- next header
			 * the one whose content is on the current screen <-- highlight this
			 */

			for (var i = 0; i < headings.length; i++) {
				// headings[i] is current element
				// if an element is already active, then current element is not active
				// if no element is already active, then current element is active
				var currNavActive = !activeNavFound;
				/**
				 * Enter the following check up only when an active nav header is not yet found
				 * Then, check the bounding rectangle of the next header
				 * The headers that are scrolled passed will have negative bounding rect top
				 * So the first one with positive bounding rect top will be the nearest next header
				 */

				if (currNavActive && i < headings.length - 1) {
					var heading = headings[i + 1];
					var next = decodeURIComponent(heading.href.split('#')[1]);
					var nextHeader = document.getElementById(next);

					if (nextHeader) {
						var top = nextHeader.getBoundingClientRect().top;
						currNavActive = top > OFFSET;
					} else {
						console.error('Can not find header element', {
							id: next,
							heading: heading,
						});
					}
				}
				/**
				 * Stop searching once a first such header is found,
				 * this makes sure the highlighted header is the most current one
				 */

				if (currNavActive) {
					activeNavFound = true;
					headings[i].classList.add('active');
				} else {
					headings[i].classList.remove('active');
				}
			}
		}, 100);
	};

	document.addEventListener('scroll', onScroll);
	document.addEventListener('resize', onScroll);
	document.addEventListener('DOMContentLoaded', function () {
		// Cache the headings once the page has fully loaded.
		headingsCache = findHeadings();
		onScroll();
	});
})();
