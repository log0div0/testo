
import React from 'react'
import Layout from './components/Layout'

module.exports = function() {
	return (
		<Layout>
			<h1>Страница покупки лицензии</h1>
			<p>Опять же, мне нравится, как сделано у <a href="https://www.sublimehq.com/store/text">sublime</a></p>
			<ul>
				<li>Заполняешь коротенькую формочку (имя, email)</li>
				<li>Нажимаешь кнопку "Оплатить"</li>
				<li>Тебя перебрасывает на Яндекс.Кассу</li>
				<li>Там ты вводишь данные карточки</li>
				<li>По окончании оплаты тебя перебрасывает обратно на наш сайт, где можно скачать лицензию
					и написано, что дальше делать с этой лицензией</li>
			</ul>
			<p>Кроме того, здесь должна быть ссылочка на <a href="/license">лицензию</a></p>
		</Layout>
	)
}
