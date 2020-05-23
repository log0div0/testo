# Часть 6. Доступ в Интернет из виртуальной машины

## С чем Вы познакомитесь

В этой части вы:

1. Познакомитесь с виртуальными сетями и сетевыми адаптерами
2. Научитесь обеспечивать доступ в Интернет внутри ваших тестов

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [пятой части](5_params).

## Вступление

Помимо виртуальных машин в платформе Testo существуют также другие виртуальные сущности: виртуальные флешки и виртуальные сети. Виртуальные флешки мы разберём позже, а в этой части сосредоточимся на виртуальных сетях.

Вообще, виртуальные сети в платформе Testo можно использовать для двух целей: связь нескольких виртуальных машин между собой и получение доступа в Интернет с виртуальной машины. Связь нескольких машин между собой мы рассмотрим в следующем уроке, а в этом познакомимся с доступом в Интернет.

## С чего начать?

Итак, в настоящий момент у нас имеется набор тестов с установкой ОС Ubuntu Server и гостевых дополнений на виртуальную машину `my_ubuntu`. Как мы знаем, зачастую "голая" Ubuntu Server не представляет очень большой ценности и на неё необходимо устанавливать дополнительное ПО, в том числе и для проведения тестов. Конечно, можно заранее подготовить всё необходимое ПО и копировать его в виртуальную машину с помощью  `copyto`, но зачастую гораздо удобнее воспользоваться пакетным репозиторием Ubuntu. Но для этого, как мы понимаем, необходим доступ в Интернет.

Для того, чтобы подключить виртуальную машину к Инетрнету, необходимо для начала в тестовых сценариях объявить виртуальную сеть, которая будет для этого использоваться. Для этого существует директива [`network`](/docs/lang/network)

```testo
network internet {
	mode: "nat"
}
```

Объявление виртуальной сети похоже на объявление виртуальной машины: после директивы `network` необходимо указать имя сети (должно быть уникальным), а также указать набор атрибутов, из которых обязательный только один: `mode` (режим работы). Существует всего два режима работы для сети: `nat` (для доступа во внешнюю сеть хоста, то есть в Интернет) и `internal` (внутренняя сеть, только для связи между несколькими виртуальными машинами).

Теперь, после объявления самой виртуальной сети, необходимо добавить сетевой адаптер в машину `my_ubuntu`, который будет подключен к этой самой сети. Для этого нам потребуется новый атрибут `nic` в объявлении машины `my_ubuntu`

```testo
machine my_ubuntu {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}
}
```

Атрибут `nic`, в отличие от других атрибутов, должден иметь имеет имя (уникальное в пределах виртуальной машины). Это связано с тем, что сетевых адаптеров в виртуальной машине может быть несколько и мы должны быть в состоянии отличать их друг от друга.

Помимо имени в сетевом адаптере также нужно указывать атрибуты. Среди них обязатльный только один - `attached_to`, который указывает, к какой виртуальной сети должен быть подключен адаптер. В нашем случае это сеть `internet`.

> Обратите внимание, что сеть `internet` должна быть уже объявлена на момент её использования в сетевом адаптере `nic`

## Подправляем тест ubuntu_installation

Давайте запустим наш тестовый сценарий

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="">Some tests have lost their cache:<br/></span>
	<span className="">	- ubuntu_installation<br/></span>
	<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">check_internet<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
	<span className="blue ">[  0%] Creating virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Taking snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
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
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install Ubuntu Server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose the language </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your location </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Detect keyboard layout? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Country of origin for the keyboard </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">No network interfaces detected </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:34:3: Error while performing action wait No network interfaces detected timeout 5m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
	<span className="red bold">[ 33%] Test </span>
	<span className="yellow bold">ubuntu_installation</span>
	<span className="red bold"> FAILED in 0h:5m:12s<br/></span>
	<span className="">user$ </span>
</Terminal>

Если ваши тесты были на этот момент закешированы, то вы увидите сообщение о том, что тест `ubuntu_installation` потерял кеш и его необходимо запустить заново. Это может показаться странным, ведь мы не трогали этот тест и его целостность не должна была потеряться. Однако, а прошлой части мы упоминали, что в целостность теста также входит целостность конфигурации всех виртуальных машин, которые в нём задействованы. Т.к. мы изменили конфигурацию виртуальной машины `my_ubuntu`, все тесты с её участием теряют актуальность и их необходимо прогнать заново.

В любом случае спустя какое-то время вы увидите, что наш тест `ubuntu_installation` перестал проходить. Вывод подсказывает нам, что Testo не смогло дождаться надписи "No network interfaces detected" в течение 5 минут.

Действительно, если мы с помощью `virtual manager` зайдём посмотреть, что происходит с нашей виртуальной машиной, мы увидим экран

![Hostname](/static/tutorials/6_nat/hostname.png)

Это произошло потому, что мы добавили сетевой интерфейс и теперь при установке Ubuntu Server больше не возникает предупреждение о том, что сетевых адаптеров не найдено.

Давайте закомментируем пока строчку с ожиданием этой надписи в установочном скрипте. Правда, теперь 30 секунд может не хватить для появления надписи "Hostname", поэтому увеличим это ожидание до 5 минут

```testo
...
wait "Country of origin for the keyboard"; press Enter
wait "Keyboard layout"; press Enter
#wait "No network interfaces detected" timeout 5m; press Enter
wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
wait "Full name for the new user"; type "${login}"; press Enter
wait "Username for your account"; press Enter
...
```

И запустим скрипт заново.

Если у вас нет проблем с прокси-сервером (или вы его вовсе не используете), то тестовый сценарий может снова сломаться.

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">check_internet<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
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
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install Ubuntu Server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose the language </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your location </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Detect keyboard layout? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Country of origin for the keyboard </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Hostname: </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">BACKSPACE </span>
	<span className="blue ">36 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"my-ubuntu" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Full name for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"my-ubuntu-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Username for your account </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose a password for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Re-enter password to verify </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Use weak password? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Encrypt your home directory? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your timezone </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:43:3: Error while performing action wait Select your timezone timeout 2m on virtual machine my_ubuntu:<br/>	-Timeout<br/></span>
	<span className="red bold">[ 33%] Test </span>
	<span className="yellow bold">ubuntu_installation</span>
	<span className="red bold"> FAILED in 0h:2m:57s<br/></span>
	<span className="">user$ </span>
</Terminal>

Вместо экрана с предложением выбрать часовой пояс мы увидим другой экран

![Timezone](/static/tutorials/6_nat/timezone.png)

Это произошло потому что благодаря Интернету установщик Ubuntu Server может автоматически определить текущиий часовой пояс. Поэтому еще немного подкорректируем тестовый сценарий

```testo
...
wait "Re-enter password to verify"; type "${password}"; press Enter
wait "Use weak password?"; press Left, Enter
wait "Encrypt your home directory?"; press Enter

#wait "Select your timezone" timeout 2m; press Enter
wait "Is this time zone correct?" timeout 2m; press Enter
wait "Partitioning method"; press Enter
...
```

Теперь все тесты должны успешно пройти.

Для проверки того, что у нас внутри тестов действительно теперь есть доступ в Интернет, давайте переименуем тест `guest_additions_demo` в `check_internet` и переделаем его таким  образом

```testo
test check_internet: guest_additions_installation {
	my_ubuntu {
		exec bash "apt update"
	}
}
```

Если запустить наш тестовый сценарий, то вывод будет следующим

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">check_internet<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">ubuntu_installation<br/></span>
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
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install Ubuntu Server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose the language </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your location </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Detect keyboard layout? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Country of origin for the keyboard </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Hostname: </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">BACKSPACE </span>
	<span className="blue ">36 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"my-ubuntu" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Full name for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"my-ubuntu-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Username for your account </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose a password for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Re-enter password to verify </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Use weak password? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Encrypt your home directory? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Is this time zone correct? </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Partitioning method </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select disk to partition </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Write the changes to disks and configure LVM? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Amount of volume group to use for guided partitioning </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Write the changes to disks? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">HTTP proxy information </span>
	<span className="blue ">for 3m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">How do you want to manage upgrades </span>
	<span className="blue ">for 6m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose software to install </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install the GRUB boot loader to the master boot record? </span>
	<span className="blue ">for 10m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Installation complete </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Unplugging dvd </span>
	<span className="yellow "> </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"my-ubuntu-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Taking snapshot </span>
	<span className="yellow ">ubuntu_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[ 33%] Test </span>
	<span className="yellow bold">ubuntu_installation</span>
	<span className="green bold"> PASSED in 0h:5m:16s<br/></span>
	<span className="blue ">[ 33%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 33%] Running test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 33%] Plugging dvd </span>
	<span className="yellow ">/opt/iso/testo-guest-additions.iso </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Typing </span>
	<span className="yellow ">"sudo su" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Waiting </span>
	<span className="yellow ">password for my-ubuntu-login </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Waiting </span>
	<span className="yellow ">root@my-ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Typing </span>
	<span className="yellow ">"mount /dev/cdrom /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Waiting </span>
	<span className="yellow ">mounting read-only </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Typing </span>
	<span className="yellow ">"dpkg -i /media/testo-guest-additions*" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Waiting </span>
	<span className="yellow ">Setting up testo-guest-additions </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Typing </span>
	<span className="yellow ">"umount /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Sleeping in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> for 2s<br/></span>
	<span className="blue ">[ 33%] Unplugging dvd </span>
	<span className="yellow "> </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 33%] Taking snapshot </span>
	<span className="yellow ">guest_additions_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[ 67%] Test </span>
	<span className="yellow bold">guest_additions_installation</span>
	<span className="green bold"> PASSED in 0h:0m:15s<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">check_internet<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">check_internet<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ apt update<br/></span>
	<span className=" "><br/></span>
	<span className=" ">WARNING: apt does not have a stable CLI interface. Use with caution in scripts.<br/></span>
	<span className=" "><br/></span>
	<span className=" ">Hit:1 http://security.ubuntu.com/ubuntu xenial-security InRelease<br/></span>
	<span className=" ">Hit:2 http://us.archive.ubuntu.com/ubuntu xenial InRelease<br/></span>
	<span className=" ">Hit:3 http://us.archive.ubuntu.com/ubuntu xenial-updates InRelease<br/></span>
	<span className=" ">Hit:4 http://us.archive.ubuntu.com/ubuntu xenial-backports InRelease<br/></span>
	<span className=" ">Reading package lists...<br/></span>
	<span className=" ">Building dependency tree...<br/></span>
	<span className=" ">Reading state information...<br/></span>
	<span className=" ">150 packages can be upgraded. Run 'apt list --upgradable' to see them.<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">check_internet</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">check_internet</span>
	<span className="green bold"> PASSED in 0h:0m:7s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:5m:39s<br/></span>
	<span className="blue bold">UP-TO-DATE: 0<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

В выводе действия `exec bash` мы явно видим успешное выполнение bash-команды `apt update`. А это означает, что внутри теста мы успешно связались с Интернетом!

## Итоги

Платформа Testo предоставляет возможность использовать связь с Интернетом внутри тестовых сценариев. Для этого существуют виртуальные сети, которые помимо связи с Интернетом также могут использоваться для связи виртуальных машин между собой. Вопрос связи нескольких виртуальных машин между собой мы рассмотрим в следующем уроке.

Итоговый файл с тестовыми сценариями можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/6)

