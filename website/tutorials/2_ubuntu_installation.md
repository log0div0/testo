# Часть 2. Устанавливаем Ubuntu Server

## С чем Вы познакомитесь

В этом первом уроке вы познакомитесь c основными действиями для виртуальных машин: `wait`, `type`, `press`

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
4. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
5. (Рекомендовано) Проделаны шаги из [первой части](1_creating_vm).

## Вступление

В прошлой части мы остановились на том, что смогли успешно объявить виртуальную машину `my_ubuntu` и смогли написать тест, в ходе которого эта виртуальная машина создается и запускается. Но возникает закономерный вопрос, что же делать с этой виртуальной машиной дальше? Как автоматизировать установку Ubuntu Server 16.04?

Как уже упоминалось, платформа Testo направлена на имитацию действий человека при работе с компьютером. При написании тестовых сценариев вам доступен весь арсенал действий, который бы делал обычный человек, сидя перед монитором, клавиатурой и мышкой.

Давайте рассмотрим пример. После запуска виртуальной машины пользователь (человек) видит перед собой следующий экран

![Убунту запущена](/static/tutorials/2_ubuntu_installation/ubuntu_started.png)

Когда человек видит этот экран, он понимает, что настало время предпринять какие-то действия. В данном случае, он понимает, что необходимо нажать клавишу Enter. После этого он дожидается следующего экрана

![Убунту запущена 2](/static/tutorials/2_ubuntu_installation/ubuntu_started_2.png)

После чего пользователь снова нажимает Enter и дожидается следующего экрана... Процесс будет повторяться до полной установки Ubuntu Server

Если подумать, работу пользователя (человека) с компьютером можно разделить на две составляющие:
1. Ожидание наступления какого-либо события (анализ происходящего на экране)
2. Реакция на это событие (нажатие на кнопки на клавиатуре)

В основе языка Testo-lang как раз и лежит автоматизация и формализация такого поведения.

## С чего начать?

Давайте рассмотрим это на примере и вернемся к текстовому сценарию, который мы написали в предыдущем уроке

```testo
machine my_ubuntu {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "/opt/iso/ubuntu_server.iso"
}

test my_first_test {
	my_ubuntu {
		start
		abort "Stop here"
	}
}
```

После запуска виртуальной машины мы видим экран с просьбой выбрать нужный язык, после чего нам необходимо нажать Enter (т.к. нас устраивает английский язык)

Для того, чтобы убедиться, что перед нами действительно нужный экран, в языке Testo-lang существует действие [`wait`](/docs/lang/actions#wait)

Действие `wait` позволяет дождаться появления определенной надписи (или комбинации надписей) на экране. Для того, чтобы убедиться, что перед нами действительно экран выбора языка, достаточно убедиться, что на экране есть надпись "English"

```testo
test my_first_test {
	my_ubuntu {
		start
		wait "English"
		abort "Stop here"
	}
}
```

Такое действие `wait` вернет управление только тогда, когда на экране появится соответствующая надпись. Давайте попробуем запустить такой тест


<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">my_first_test<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:13:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">my_first_test</span>
	<span className="red bold"> FAILED in 0h:0m:4s<br/></span>
	<span className="">user$ </span>
</Terminal>

Заметим, что действие `abort` все еще присутствует, выполняя роль своеобразного break point в нашем тестовом сценарии. Так процесс создания скрипта становится максимально наглядным, т.к. мы сразу же можем увидеть состояние виртуальной машины, в котором она оказывается в момент вызова `abort`

> Действие `wait` отлично работает и с русскими буквами. Вместо `wait "English"` можно было бы искать `wait "Русский"`

Теперь, когда в тестовом сценарии мы убедились, что перед нами действительно экран выбора языка, можно приступать к реагирование на наступление этого события. В нашем случае необходимо нажать клавишу Enter чтобы выбрать английский язык. Для нажатия клавиш в языке testo-lang есть действие [`press`](/docs/lang/actions#press)

```testo
test my_first_test {
	my_ubuntu {
		start
		wait "English"
		press Enter
		abort "Stop here"
	}
}
```

Вывод:

<Terminal height="350px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">my_first_test<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:13:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">my_first_test</span>
	<span className="red bold"> FAILED in 0h:0m:4s<br/></span>
	<span className="">user$ </span>
</Terminal>

Если открыть виртуальную машину `my_ubuntu` в `virtual manager`, то мы увидим, что установка Убунту, действильно, немного сдвинулась с места: теперь перед нами второй экран установки с выбором необходимых действий.

По аналогии, нам необходимо для начала убедиться, что перед нами действительно нужный экран. Это можно сделать, выполнив действие `wait "Install Ubuntu Server"`. После чего можно снова нажимать Enter

Таким образом, процесс создания тестового сценария можно подытожить как набор связок "Дождаться события, прореагировать на событие". Выглядит это примерно так:

![Action flow](/static/tutorials/2_ubuntu_installation/action_flow.png)

## wait timeout

Как уже упоминалось, действие `wait` не сработает до тех пор, пока на экрнане не появится соответствующая надпись. Но что делать, если надпись вообще никогда не появится? Такое может быть, например, если мы тестируем программу и ожидаем увидеть надпись "Успешно", а программа работает при этом неправильно и вместо "Успешно" появляется "Ошибка". В этом случае действие `wait` не будет работать вечно, т.к. у него есть таймаут, который по умолчанию равен одной минуте.

```testo
test my_first_test {
	my_ubuntu {
		start
		wait "English" timeout 1m #Это эквивалентно действию wait "English"
		press Enter
		abort "Stop here"
	}
}
```

Для обозначения временных интервалов в Testo-lang существуют [специальные литералы](/docs/lang/lexems#спецификатор-количества-времени)(как и для количества памяти).

Давайте попробуем убедиться, что команда `wait` действительно не передаст управление дальше пока на экране действительо не появится нужная надпись. Попробуем вместо "English" искать какую-нибудь тарабарщину. А чтобы не ждать одну минуту, сделаем таймаут 10 секунд

```testo
test my_first_test {
	my_ubuntu {
		start
		wait "ALALA" timeout 10s
		press Enter
		abort "Stop here"
	}
}
```

Вывод:

<Terminal height="350px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">my_first_test<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">my_first_test<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">ALALA </span>
	<span className="blue ">for 10s with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:13:3: Error while performing action wait ALALA timeout 10s on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">my_first_test</span>
	<span className="red bold"> FAILED in 0h:0m:11s<br/></span>
	<span className="">user$ </span>
</Terminal>

Видно, что ошибка произошла уже не в действии `abort`, а пораньше, в действии `wait`

## type

Если продолжить написание тестового сценария, то на этом этапе

```testo
test my_first_test {
	my_ubuntu {
		start
		wait "English";
		press Enter;
		wait "Install Ubuntu Server"; press Enter;
		wait "Choose the language";	press Enter
		wait "Select your location"; press Enter
		wait "Detect keyboard layout?";	press Enter
		wait "Country of origin for the keyboard"; press Enter
		wait "Keyboard layout"; press Enter
		wait "No network interfaces detected" timeout 5m

		#Обратите внимание, при необходимости нажатия нескольких клавиш подряд
		#их можно объединить в одном действии press с помощью запятой
		press Right, Enter
		wait "Hostname:"
		abort "Stop here"
	}
}
```

Мы увидим следующий экран, в котором нам предлагаю выбрать имя хоста для нашей виртуальной машины

![Hostname](/static/tutorials/2_ubuntu_installation/hostname.png)

Конечно, можно было бы оставить значение по-умолчанию, но что, если мы хотим ввести другое значение?

Для этого нам необходимо выполнить два шага:
1. Стереть существующее значение.
2. Ввести новое значение.

Для того, чтобы стереть существующее значение, нам необходимо нажать клавишу Backspace как минимум 6 раз. Для того чтобы не пришлось дублировать действие `press` 6 раз, можно написать так: `press Backspace*6`

Теперь нужно разобраться с набором нового имени хоста (допустим, нам устраивает имя `my-ubuntu`). Конечно, можно было бы набрать эту строку через действие `press` (поочередно выбирая клавиши, которые нужно нажать), но для набора текста в testo-lang существует отдельная команда [`type`](/docs/lang/actions#type)

Для набора нового имени достаточно написать действие `type "my-ubuntu"`

Это же действие `type` позволит разобраться с вводом логина (для примера можно сделать логин `my-ubuntu-login`) и пароля (`1111`)

> Дейтсвие `type` также работает вместе с русскими надписями. Например, можно написать `type "Привет, мир!"`. Однако на виртуальной машине в это время должна быть включена русская раскладка клавиатуры. Попытка ввести русский текст на включенной английской раскладке приведет к нежелательному результату. Раскладку клавиатуры можно поменять, например, с помощью `press LeftShift + LeftAlt`

## Завершение установки

После окончания всех необходимых установочных действий мы увидим экран с завершением установки, в котором нам предложат вытащить установочный диск и нажать Enter

![Installation Complete](/static/tutorials/2_ubuntu_installation/installation_complete.png)

Платформа Testo позволяет не только имитировать ввод данных с помощью клавиатуры, но и управление оборудованием виртуальной машины. Для подключения-отключения различных устройств в виртуальной машине существует действия [`plug`](/docs/lang/actions#plug) и [`unplug`](/docs/lang/actions#unplug). В полной меере с использвоанием этих действий мы будем знакомиться постепенно в ходе дальнейших уроков, а сейчас сосредоточимся на том, как вытащить установочный диск из дисковода виртуальной машины. Для этого необходимо выполнить действие `unplug dvd`.

После этого останется только дождаться завершения перезагрузки (появится экран с предложение ввести данные учётной записи) и выполнить логин

```testo
test my_first_test {
	my_ubuntu {
		...

		wait "Installation complete" timeout 1m;
		unplug dvd; press Enter
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

Если все все действия отрабатывают успешно, можно убрать в конце точку останова `abort "stop here"`	 и, таким образом, завершить тест `my_first_test`.

Поздравляем! Вы написали тестовый сценарий, который в полностью автоматическом режиме устанавливает на "пустую" виртуальную машину ОС Ubuntu Server 16.04!

## Готовый тестовый скрипт

Итоговый тестовый скрипт можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/2)

> Внимание! Шаги тестового сценария могут меняться если у вас другие настройки сети (в частности, если вы используете HTTP-прокси). При необходимости подкорректируйте предложенный скрипт.