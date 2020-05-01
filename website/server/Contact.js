
import React from 'react'
import Layout from './components/Layout'

module.exports = function() {
	return (
		<Layout>
			<h1>Форма обратной связи</h1>
			<p>Примерное содержание:</p>
			<ul>
				<li>Как к Вам обращаться?</li>
				<li>Ваш email</li>
				<li>Компания/должность?</li>
				<li>Чем мы можем Вам помочь?</li>
			</ul>
			<p>Я вообще не уверен, что форма обратной связи нужна. Может ограничится нашим email на странице <a href="/help">Помощь</a>?</p>
		</Layout>
	)
}
