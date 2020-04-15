
var coll = document.getElementsByClassName('collapsible');
for (var i = 0; i < coll.length; i++) {
	coll[i].addEventListener('click', function() {
		var arrow = this.childNodes[1];
		arrow.classList.toggle('rotate');
		var content = this.nextElementSibling;
		content.classList.toggle('hide');
	});
}
