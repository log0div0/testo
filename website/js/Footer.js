
import React from 'react'

module.exports = function() {
	return (
		<div className="nav-footer">
			<section className="sitemap">
				<div>
					<a href="/" className="nav-home">
						<img src="/static/logo_v.svg" alt="Testo Lang"/>
					</a>
				</div>
				<div>
					<h5>Контакты</h5>
					<a href="#">Ссылка 1</a>
					<a href="#">Ссылка 2</a>
					<a href="#">Ссылка 3</a>
					<a href="#">Ссылка 4</a>
				</div>
				<div>
					<h5>Помощь</h5>
					<a href="#">Ссылка 1</a>
					<a href="#">Ссылка 2</a>
					<a href="#">Ссылка 3</a>
				</div>
				<div>
					<h5>Подписывайтесь на нас</h5>
					<div className="social">
						<script src="https://apis.google.com/js/platform.js"></script>
						<div className="g-ytsubscribe" data-channelid="UC4voSBtFRjRE4V1gzMZoZuA" data-layout="default" data-count="default"></div>
					</div>
				</div>
			</section>
		</div>
	)
}
