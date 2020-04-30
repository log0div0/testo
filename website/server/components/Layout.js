
import React from 'react'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
				<link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet"/>
				<link rel="stylesheet" href="/main.css" type="text/css"/>
			</head>
			<body>
				<header>
					<nav>
						<section>
							<a href="/">
								<h1>Testo Lang</h1>
							</a>
						</section>
						<section>
							<a href="/downloads">Загрузить</a>
							<a href="/buy">Купить</a>
							<a href="/features">Возможности платформы</a>
							<a href="/examples">Посмотреть в действии</a>
							<a href="/tutorials/1_creating_vm">Обучающие материалы</a>
							<a href="/docs/getting_started/intro">Документация</a>
							<a href="/contact">Связаться с нами</a>
							<a href="/help">Помощь</a>
						</section>
					</nav>
				</header>
				<main>
					{children}
				</main>
				<footer>
					<nav>
					</nav>
				</footer>
			</body>
			<script type="text/javascript" src="/static/client.js"></script>
		</html>
	)
}
