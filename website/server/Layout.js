
import React from 'react'
import Header from './Header'
import Footer from './Footer'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
				<link rel="stylesheet" href="/main.css" type="text/css"/>
				<link rel="stylesheet" href="/static/css/terminal.css" type="text/css"/>
				<link rel="stylesheet" href="/static/css/hljs/monokai-sublime.css" type="text/css"/>
			</head>
			<body>
				<Header/>
				{children}
				<Footer/>
			</body>
			<script type="text/javascript" src="/static/js/main.js"></script>
		</html>
	)
}
