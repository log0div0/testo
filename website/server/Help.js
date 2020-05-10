

import React from 'react'
import Layout from './components/Layout'

module.exports = function() {
	return (
		<Layout>
			<h1>Помощь</h1>
			<p>Для технических вопросов мы могли бы указать здесь ссылочку на stackoverflow</p>
			<p>Для бактрекинга мы могли бы использовать раздел issues в нашем публичном репозитории на github
				(я видел 2 компании, которые так делают)</p>
			<p>Можно здесь же указать ссылку на youtube канал и на другие медиа, если они появятся</p>
			<p>Можно указать здесь ссылочку на технический FAQ (или прям здесь вывесить FAQ)</p>
			<p>Ну и в самом конце - наш email: support@testo-lang.ru</p>

			<h1>Форма обратной связи</h1>
			<p>Примерное содержание:</p>
			<ul>
				<li>Как к Вам обращаться?</li>
				<li>Ваш email</li>
				<li>Компания/должность?</li>
				<li>Чем мы можем Вам помочь?</li>
			</ul>

			<h1>FAQ</h1>
			<ol>
				<li><a href="#what-is-testo">Что такое платформа Testo?</a></li>
				<li><a href="#profits">Какую пользу может принести мне платформа Testo?</a></li>
				<li><a href="#what-you-can-automate">Какие тесты можно автоматизировать на платформе Testo?</a></li>
				<li><a href="#how-much">Насколько беслпатна платформа Testo?</a></li>
				<li><a href="#system-requirements">Какие системные требования у платформы Testo?</a></li>
				<li><a href="#hypervisors">Какие гипервизоры поддерживает платформа Testo?</a></li>
				<li><a href="#how-to-begin">Я хочу начать пользоваться платформой Testo. С чего мне начать?</a></li>
				<li><a href="#tutorials-not-enough">Я посмотрел все туториалы на сайте и всё равно не понимаю, как реализовать тестовый сценарий. Что мне делать?</a></li>
				<li><a href="#new-features">Мне не хватает возможностей Testo, я бы хотел, чтобы появилась возможность Х. Как её можно получить?</a></li>
				<li><a href="#tests-outsource">Я не хочу писать тесты сам. Может ли кто-то написать их за меня?</a></li>
				<li><a href="#nn-improvement">Testo не может распознать надпись, которая точно есть на экране. Что с этим делать?</a></li>
				<li><a href="#nn-slow">Поиск надписей на экране работает слишком медленно. Есть ли способ его ускорить?</a></li>
			</ol>

			<h3 id="what-is-testo">Что такое платформа Testo?</h3>
			<p>Testo - это платформа по автоматизации системных (комплексных) тестов. С помощью специального разработанного языка Testo-lang вы получаете возможность
			максимально понятным способом описывать тестовые сценарии с участием виртуальных машин, которые затем интерпретируются платформой Testo.
			Подробнее ознакомиться с основными преимуществами этой платформы можно <a href="features">здесь</a></p>

			<h3 id="profits">Какую пользу может принести мне платформа Testo?</h3>
			<p>Не секрет, что даже самая выверенная и оттестированная сама по себе программа может повести себя некорректно если её поместить в конкретное
			окружение: неудачная интеграция с ОС, недостаточная отказоустойчивость (например, при выключении питания), неправильная реакция на проблемы с файловой системой,
			медленная реакция на изменения в сети - вот лишь малая часть того, что может пойти не так когда вашей программой будет пользоваться конечный потребитель.
			Платформа Testo позволяет вам всегда быть уверенными, что ваша программа ведёт себя корректно в какие бы конечные условия её не поместили.</p>

			<p>Если Вы разработчик, то наверняка сталкивались с тем, что программу приходится тестировать на определенных стендах, и в связи с этим приходится
			делать много рутины: подготавливать стенды, научиться загружать сборки на этот стенд, а затем каждый раз вручную (или с помощью скриптов)
			проверять, насколько удачно работает та или иная сборка. Платформа Testo может сделать это всё за вас! Нажмите всего лишь одну кнопку
			и Testo развернет стенд (или приведет его в надлежащий вид), скопирует вашу программу на стенд, и проведет все необходимые проверки полностью автоматически.</p>

			<h3 id="what-you-can-automate">Какие тесты можно автоматизировать на платформе Testo?</h3>
			<p>Платформа Testo предназначена для автоматизации системных тестов на основе взаимодействия виртуальных машин. При этом Testo позволяет
			Вам писать тестовые сценарии с действиями, которые имитируют действия человека. Таким образом, вы можете автоматизировать любые тесты,
			которые может выполнить вручную человек, сидя за компьютером (и имеющий возможность перемещаться между несколькими компьютерами).
			Причём под действиями человека понимаются не только работа с клавиатурой или мышкой, но и работа с "железом" - вставка/извлечение флешек,
			управление питанием компьютера, сетевыми кабелями, cd-приводом и пр. Примеры таких тестов вы можете посмотреть здесь. </p>

			<h3 id="how-much">Насколько беслпатна платформа Testo?</h3>
			<p>Платформа Testo бесплатна для некоммерческого использования с распознаванием образов на экране с помощью CPU. При необходимости следующих возможностей
			необходимо приобрести лицензию:</p>
			<ul>
				<li>Использование Testo в коммерческих проектах (Стандартная лицензия)</li>
				<li>Техническая поддержки с приоритетным исправлением багов и консультациями по использовании платформы Testo (Стандартная лицензия)</li>
				<li>Использование GPU для существенного ускорения распознавания образов на экране (Стандартная лицензия+)</li>
				<li>Улучшение точности нейросетей по требованию (Расширенная лицензия)</li>
				<li>Улучшение точности нейросетей по требованию при использовании GPU (Расширенная лицензия+)</li>
				<li></li>
			</ul>

			<h3 id="system-requirements">Какие системные требования у платформы Testo?</h3>
			<p></p>
			<ol>
				<li>CPU:</li>
				<li>RAM:</li>
				<li>Ubuntu 18.04 и старше</li>
				<li>Совместимый GPU-адаптер при использовании распознавания на графических процессорах</li>
			</ol>

			<p>Testo может работать практически с любыми основанными на Linux операционными системами. Если Вам требуется портировать Testo на новую
			операционную систему - напишите нам.</p>

			<h3 id="hypervisors">Какие гипервизоры поддерживает платформа Testo?</h3>
			<p>В настоящий момент полноценно поддерживается только работа с гипервизором QEMU/KVM, в будущем планируется добавить поддержку гиперизора Hyper-V
			компании Microsoft</p>

			<h3 id="how-to-begin">Я хочу начать пользоваться платформой Testo. С чего мне начать?</h3>
			<p>Для начала скачайте и установите бесплатную версию платформы Testo. Ознакомиться с её возможностями можно в  
			<a href="/tutorials/1_creating_vm"> обучающих материалах</a>. Если Вы планируете использовать Testo в коммерческих
			проектах или вам требуется поддержка распознавания на GPU - приобретите соответствующюю лицензию.
			При возникновении любых вопросов пишите нам.</p>

			<h3 id="tutorials-not-enough">Я посмотрел все туториалы на сайте и всё равно не понимаю, как реализовать тестовый сценарий. Что мне делать?</h3>
			<p>
				Задать нам вопрос касательно построения тестовых сценариев можно на StackOverflow. Также вы можете написать напрямую нам на почту.
				Если вопрос выходит за рамки простой консультации по синтаксису и конструкциям языка, то для получения комплексной консультации
				Вам может потребоваться приобрести лицензию.
			</p>

			<h3 id="new-features">Мне не хватает возможностей Testo, я бы хотел, чтобы появилась возможность Х. Как её можно получить?</h3>
			<p>
				Мы стараемся учитывать пожелания наших пользователей при планировании развития Testo. Если вам кажется, что в платформе Testo
				очень не хватает какой-либо возможности - напишите нам, мы постараемся её имплементировать в будущих релизах.
			</p>

			<p>
				Если Вам хочется, чтобы возможность Х появилась в Testo как можно быстрее, то вы можете стать нашим партнёром - в этом случае
				мы отдадим предпочтение при разработке Testo именно этой нехватающей возможности. По этим вопросам, пожалуйста, напишите нам на почту.
			</p>

			<h3 id="tests-outsource">Я не хочу писать тесты сам. Может ли кто-то написать их за меня?</h3>
			<p>
				Мы предоставляем услугу создания тестов за отдельную плату. Каждое обращение рассматривается индивидуально.
				Если вы заинтересованы в создании тестов "под ключ" - напишите нам на почту.
			</p>

			<h3 id="nn-improvement">Testo не может распознать надпись, которая точно есть на экране. Что с этим делать?</h3>
			<p>
				Во-первых, убедитесь, что вашу надпись ничто не загораживает. В ходе тестовых сценариев бывает, что искомая надпись
				частично перекрывается курсором мышки, из-за чего её не получается распознать. Если дело действительно в этом, то
				достаточно переместить курсор мыши куда-нибудь в сторону. 
			</p>

			<p>
				В остальных случаях следует помнить, что механизм распознавания образов на экране основан на работе нейросетей.
				Иногда даже самые хорошо обученные нейросети сталкиваются с ситуацией, когда они не могут справиться с задачей
				по распознаванию нужного образа на экране. Если вы столкнулись с такой ситуацией, алгоритм действий следующий:
			</p>

			<ol>
				<li>Во-первых, напишите нам. Приложите к сообщению скриншот экрана виртуальной машины, а также надпись, которая должна была распознаться</li>
				<li>Если Вы обладаете расширенной лицензией или расширенной лицензией+, мы решим проблему с распознаванием в ближайшее время и вышлем Вам исправленную сборку в кратчайшие сроки</li>
				<li>В остальных случаях мы примем к сведению недоработку нейросетей и постараемся дообучить нейросеть к следующему релизу платформы Testo</li>
				<li>Как временное решение - попробуйте искать на экране другую надпись или подстроку в исходной надписи. С большой долей вероятности этот приём должен сработать</li>
			</ol>

			<h3 id="nn-slow">Поиск надписей на экране работает слишком медленно. Есть ли способ его ускорить?</h3>
			<p>
				По умолчанию платформа Testo запускает механизм распознавания образов с экрана на CPU. Скорость таких распознаваний зависит
				от разрешения экрана и от количества надписей на экране. Вы можете попробовать ускорить распознавание, уменьшив разрешение экрана.
			</p>

			<p>
				Но при этом в любом случае распознавание образов на CPU будет проигрывать в скорости распознаванию на GPU. Вы можете задействовать
				распознавание образов на GPU в платформе Testo, купив Стандартную лицензию+. Этот режим также позволяет вам экономить ресурс CPU.
			</p>
		</Layout>
	)
}