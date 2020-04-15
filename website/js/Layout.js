
import React from 'React'
import Header from './Header'

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
				<script type="text/javascript" src="/client.js"></script>
			</body>
		</html>
	)
}
