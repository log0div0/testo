
import React from 'React'
import Header from './Header'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
				<link rel="stylesheet" href="/main.css"/>
			</head>
			<body className="sideNavVisible separateOnPageNav">
				<Header/>
				<div className="navPusher">
					{children}
				</div>
			</body>
		</html>
	)
}
