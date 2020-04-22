# Часть 6. Доступ в Интернет из виртуальной машины

## С чем Вы познакомитесь

В этой части вы:

1. Познакомитесь с виртуальными сетями и сетевыми адаптерами
2. Научитесь обеспечивать доступ в Интернет внутри ваших тестов

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствуующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [пятой части](5_params).

## Вступление

Помимо виртуальных машин в платформе Testo существуют также другие виртуальные сущности: виртуальные флешки и виртуальные сети. Виртуальные флешки мы разберём позже, а в этой части сосредоточимся на виртуальных сетях.

Вообще, виртуальные сети в платформе Testo можно использовать для двух целей: связь нескольких виртуальных машин между собой и получение доступа в Интернет с виртуальной машины. Связь нескольких машин между собой мы рассмотрим в следующем уроке, а в этом познакомимся с доступом в Интернет.

## С чего начать?

Итак, в настоящий момент у нас имеется набор тестов с установкой ОС Ubuntu Server и гостевых дополнений на виртуальную машину `my_ubuntu`. Как мы знаем, зачастую "голая" Ubuntu Server не представляет очень большой ценности и на неё необходимо устанавливать дополнительное ПО, в том числе и для проведения тестов. Конечно, можно заранее подготовить всё необходимое ПО и копировать его в виртуальную машину с помощью  `copyto`, но зачастую гораздо удобнее воспользоваться пакетным репозиторием Ubuntu. Но для этого, как мы понимаем, необходим доступ в Интернет.

Для того, чтобы подключить виртуальную машину к Инетрнету, необходимо для начала в тестовых сценариях объявить виртуальную сеть, которая будет для этого использоваться. Для этого существует директива [`network`](/docs/lang/network)

	network internet {
		mode: "nat"
	}
	
Объявление виртуальной сети похоже на объявление виртуальной машины: после директивы `network` необходимо указать имя сети (должно быть уникальным), а также указать набор атрибутов, из которых обязательный только один: `mode` (режим работы). Существует всего два режима работы для сети: `nat` (для доступа во внешнюю сеть хоста, то есть в Интернет) и `internal` (внутренняя сеть, только для связи между несколькими виртуальными машинами).

Теперь, после объявления самой виртуальной сети, необходимо добавить сетевой адаптер в машину `my_ubuntu`, который будет подключен к этой самой сети. Для этого нам потребуется новый атрибут `nic` в объявлении машины `my_ubuntu`

	machine my_ubuntu {
		cpus: 1
		ram: 512Mb
		disk_size: 5Gb
		iso: "${ISO_DIR}/ubuntu_server.iso"

		nic nat: {
			attached_to: "internet"
		}
	}

Атрибут `nic`, в отличие от других атрибутов, должден иметь имеет имя (уникальное в пределах виртуальной машины). Это связано с тем, что сетевых адаптеров в виртуальной машине может быть несколько и мы должны быть в состоянии отличать их друг от друга.

Помимо имени в сетевом адаптере также нужно указывать атрибуты. Среди них обязатльный только один - `attached_to`, который указывает, к какой виртуальной сети должен быть подключен адаптер. В нашем случае это сеть `internet`.

> Обратите внимание, что сеть `internet` должна быть уже объявлена на момент её использования в сетевом адаптере `nic`

## Подправляем тест ubuntu_installation

Давайте запустим наш тестовый сценарий 

```sh
# sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso
```

Если ваши тесты были на этот момент закешированы, то вы увидите сообщение о том, что тест `ubuntu_installation` потерял кеш и его необходимо запустить заново. Это может показаться странным, ведь мы не трогали этот тест и его целостность не должна была потеряться. Однако, а прошлой части мы упоминали, что в целостность теста также входит целостность конфигурации всех виртуальных машин, которые в нём задействованы. Т.к. мы изменили конфигурацию виртуальной машины `my_ubuntu`, все тесты с её участием теряют актуальность и их необходимо прогнать заново.

В любом случае спустя какое-то время вы увидите, что наш тест `ubuntu_installation` перестал проходить.

	TESTS TO RUN:
	ubuntu_installation
	guest_additions_installation
	guest_additions_demo
	[  0%] Preparing the environment for test ubuntu_installation
	[  0%] Creating virtual machine my_ubuntu
	[  0%] Taking snapshot initial for virtual machine my_ubuntu
	[  0%] Running test ubuntu_installation
	[  0%] Starting virtual machine my_ubuntu
	[  0%] Waiting English for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Install Ubuntu Server for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Choose the language for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Select your location for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Detect keyboard layout? for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Country of origin for the keyboard for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Keyboard layout for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting No network interfaces detected for 5m with interval 1s in virtual machine my_ubuntu
	/home/alex/testo/hello_world.testo:34:3: Error while performing action wait No network interfaces detected timeout 5m on virtual machine my_ubuntu:
		-Timeout
	[ 33%] Test ubuntu_installation FAILED in 0h:5m:13s

Вывод подсказывает нам, что Testo не смогло дождаться надписи "No network interfaces detected" в течение 5 минут.

Действительно, если мы с помощью `virtual manager` зайдём посмотреть, что происходит с нашей виртуальной машиной, мы увидим экран

<br/><br/>

![Hostname](/static/tutorials/6_nat/hostname.png)

<br/><br/>

Это произошло потому, что мы добавили сетевой интерфейс и теперь при установке Ubuntu Server больше не возникает предупреждение о том, что сетевых адаптеров не найдено.

Давайте закомментируем пока строчку с ожиданием этой надписи в установочном скрипте. Правда, теперь 30 секунд может не хватить для появления надписи "Hostname", поэтому увеличим это ожидание до 5 минут

	...
	wait "Country of origin for the keyboard"; press Enter
	wait "Keyboard layout"; press Enter
	#wait "No network interfaces detected" timeout 5m; press Enter
	wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
	wait "Full name for the new user"; type "${login}"; press Enter
	wait "Username for your account"; press Enter
	...

И запустим скрипт заново.

Если у вас нет проблем с прокси-сервером (или вы его вовсе не используете), то тестовый сценарий может снова сломаться.

	TESTS TO RUN:
	ubuntu_installation
	guest_additions_installation
	guest_additions_demo
	...
	[  0%] Waiting Hostname: for 5m with interval 1s in virtual machine my_ubuntu
	...
	[  0%] Waiting Encrypt your home directory? for 1m with interval 1s in virtual machine my_ubuntu
	[  0%] Pressing key ENTER on virtual machine my_ubuntu
	[  0%] Waiting Select your timezone for 2m with interval 1s in virtual machine my_ubuntu
	/home/alex/testo/hello_world.testo:43:3: Error while performing action wait Select your timezone timeout 2m on virtual machine my_ubuntu:
		-Timeout
	[ 33%] Test ubuntu_installation FAILED in 0h:2m:57s

Вместо экрана с предложением выбрать часовой пояс мы увидим другой экран

<br/><br/>

![Timezone](/static/tutorials/6_nat/timezone.png)

<br/><br/>

Это произошло потому что благодаря Интернету установщик Ubuntu Server может автоматически определить текущиий часовой пояс. Поэтому еще немного подкорректируем тестовый сценарий

	...
	wait "Re-enter password to verify"; type "${password}"; press Enter
	wait "Use weak password?"; press Left, Enter
	wait "Encrypt your home directory?"; press Enter
	
	#wait "Select your timezone" timeout 2m; press Enter
	wait "Is this time zone correct?" timeout 2m; press Enter
	wait "Partitioning method"; press Enter
	...

Теперь все тесты должны успешно пройти.

Для проверки того, что у нас внутри тестов действительно теперь есть доступ в Интернет, давайте переименуем тест `guest_additions_demo` в `check_internet` и переделаем его таким  образом

	test check_internet: guest_additions_installation {
		my_ubuntu {
			exec bash "apt update"
		}
	}

Если запустить наш тестовый сценарий, то вывод будет следующим

	UP-TO-DATE TESTS:
	ubuntu_installation
	guest_additions_installation
	TESTS TO RUN:
	check_internet
	[ 67%] Preparing the environment for test check_internet
	[ 67%] Restoring snapshot guest_additions_installation for virtual machine my_ubuntu
	[ 67%] Running test check_internet
	[ 67%] Executing bash command in virtual machine my_ubuntu with timeout 10m
	+ apt update

	WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

	Get:1 http://security.ubuntu.com/ubuntu xenial-security InRelease [109 kB]
	Hit:2 http://us.archive.ubuntu.com/ubuntu xenial InRelease
	Hit:3 http://us.archive.ubuntu.com/ubuntu xenial-updates InRelease
	Hit:4 http://us.archive.ubuntu.com/ubuntu xenial-backports InRelease
	Fetched 109 kB in 0s (181 kB/s)
	Reading package lists...
	Building dependency tree...
	Reading state information...
	150 packages can be upgraded. Run 'apt list --upgradable' to see them.
	[ 67%] Taking snapshot check_internet for virtual machine my_ubuntu
	[100%] Test check_internet PASSED in 0h:0m:8s
	PROCESSED TOTAL 3 TESTS IN 0h:0m:8s
	UP-TO-DATE: 2
	RUN SUCCESSFULLY: 1
	FAILED: 0

В выводе действия `exec bash` мы явно видим успешное выполнение bash-команды `apt update`. А это означает, что внутри теста мы успешно связались с Интернетом!

## Итоги

Платформа Testo предоставляет возможность использовать связь с Интернетом внутри тестовых сценариев. Для этого существуют виртуальные сети, которые помимо связи с Интернетом также могут использоваться для связи виртуальных машин между собой. Вопрос связи нескольких виртуальных машин между собой мы рассмотрим в следующем уроке.

Итоговый файл с тестовыми сценариями можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/6)

