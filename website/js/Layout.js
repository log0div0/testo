
import React from 'react'
import Header from './Header'
import Footer from './Footer'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
				<link rel="stylesheet" href="/main.css" type="text/css"/>
			</head>
			<body className="sideNavVisible separateOnPageNav">
				<Header/>
				<div className="navPusher">
					{children}
				</div>
				<Footer/>
				<script type="text/javascript" src="/static/client.js"></script>
			</body>
		</html>
	)
}
