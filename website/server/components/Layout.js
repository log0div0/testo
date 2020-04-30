
import React from 'react'
import Header from './Header'
import Footer from './Footer'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
				<link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet"/>
				<link rel="stylesheet" href="/main.css" type="text/css"/>
			</head>
			<body>
				<Header/>
				<main>
					{children}
				</main>
				<Footer/>
			</body>
			<script type="text/javascript" src="/static/client.js"></script>
		</html>
	)
}
