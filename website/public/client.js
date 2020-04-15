
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
