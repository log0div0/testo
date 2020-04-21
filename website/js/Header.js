
import React from 'react'

module.exports = function() {
	return (
		<div className="fixedHeaderContainer">
			<div className="wrapper">
				<header>
					<a href="/">
						<h2>Testo Lang</h2>
					</a>
					<div className="navigationSlider">
						<nav className="slidingNav">
							<ul>
								<li>
									<a href="#">Загрузить</a>
								</li>
								<li>
									<a href="#">Купить</a>
								</li>
								<li id="nav-site-docs">
									<a href="/docs/getting_started/intro">Документация</a>
								</li>
								<li id="nav-site-tutorials">
									<a href="/tutorials/1_creating_vm">Обучающие материалы</a>
								</li>
							</ul>
						</nav>
					</div>
				</header>
			</div>
		</div>
	)
}
