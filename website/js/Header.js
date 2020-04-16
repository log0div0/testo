
import React from 'react'

module.exports = function() {
	return (
		<div className="fixedHeaderContainer">
			<div className="headerWrapper wrapper">
				<header>
					<a href="/">
						<h2 className="headerTitleWithLogo">Testo Lang</h2>
					</a>
					<div className="navigationWrapper navigationSlider">
						<nav className="slidingNav">
							<ul className="nav-site nav-site-internal">
								<li>
									<a href="#">Загрузить</a>
								</li>
								<li>
									<a href="#">Купить</a>
								</li>
								<li id="nav-site-docs">
									<a href="/docs/01/01">Документация</a>
								</li>
								<li id="nav-site-tutorials">
									<a href="/tutorials/01/01">Обучающие материалы</a>
								</li>
							</ul>
						</nav>
					</div>
				</header>
			</div>
		</div>
	)
}
