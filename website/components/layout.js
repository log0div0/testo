
import React from 'React'
import Header from '../components/header'

module.exports = function({children}) {
	return (
		<html>
			<head>
				<title>Testo Lang</title>
			</head>
			<body>
				<Header/>
				{children}
			</body>
		</html>
	)
}
