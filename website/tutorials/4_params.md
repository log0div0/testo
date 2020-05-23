# Часть 4. Параметры

## С чем Вы познакомитесь

В этой части вы познакомитесь с механизмом параметров в языке Testo lang

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
4. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
5. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
6. (Рекомендовано) Проделаны шаги из [третьей части](3_guest_additions).

## Вступление

В течение прошлых уроков мы написали довольно много кода для наших тестовых сценариев, и вы могли заметить, что в этом коде фигурирует довольно несколько фиксированных значений, которые при этом неоднократно повторяются. Например, путь для iso-образов ubuntu server и testo-guest-additions прописан явно в самом тестовом сценарии, что не очень удобно. Если iso-файлы будут располагаться в другой папке, то потребуется исправлять сам тестовый сценарий. Или, например, логин `my-ubuntu-login` тоже явно повторяется несколько раз по ходу сценария, и если бы мы захотели исправить этот логин на другое значение, то нам пришлось бы искать все вхождения этой строки в тестовом сценарии. Конечно, всё это не добавляет гибкости и лаконичности таким тестовым сценариям.

Для решения этой проблемы в языке Testo-lang существует механизм [параметров](/docs/lang/param). Параметры можно рассматривать как глобальные строковые константы. Давайте познакомимся с этим механизмом на примере.

## С чего начать?

Давайте вспомним, в каком виде находится тестовый сценарий на текущий момент.

```testo
...

test ubuntu_installation {
	my_ubuntu {
		start
		...
		wait "Hostname:" timeout 30s; press Backspace*36; type "my-ubuntu"; press Enter
		wait "Full name for the new user"; type "my-ubuntu-login"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "1111"; press Enter
		wait "Re-enter password to verify"; type "1111"; press Enter
		...
		unplug dvd; press Enter
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"
		...
	}
}

...
```

Можно заметить, что в тестовом сценарии несколько раз повторяются строковые константы `my-ubuntu`, `my-ubuntu-login` и `1111`. Понятно, что чем больше будет наш тестовый сценарий и чем больше там будет таких строковых констант, тем проще будет в них запутаться и рано или поздно допустить ошибку. Для того, чтобы избежать такой неприятной ситуации, давайте попробуем объявить несколько параметров.

```testo
param hostname "my-ubuntu"
param login "my-ubuntu-login"
param password "1111"

test ubuntu_installation {
	...
```

> Объявление параметров должно располагаться на том же уровне, что и объявление виртуальных машин или тестов (в глобальном пространстве). Нельзя объявлять параметры внутри тестов или внутри объявлений виртуальных машин и других сущностей

> Нельзя переобъявлять ранее объявленные параметры

Теперь внутри самих тестов мы можем использовать обращение к этим параметрам

```testo
...
param hostname "my-ubuntu"
param login "my-ubuntu-login"
param password "1111"

test ubuntu_installation {
	my_ubuntu {
		start
		...
		wait "Hostname:" timeout 30s; press Backspace*36; type "${hostname}"; press Enter
		wait "Full name for the new user"; type "${login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "${password}"; press Enter
		wait "Re-enter password to verify"; type "${password}"; press Enter
		...
		unplug dvd; press Enter
		wait "login:" timeout 2m; type "${login}"; press Enter
		wait "Password:"; type "${password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		#Обратите внимание, обращаться к параметрам можно в любом участке строки
		wait "password for ${login}"; type "${password}"; press Enter
		wait "root@${hostname}"
		...
	}
}

...
```

Теперь наш тестовый сценарий выглядит немного лаконичнее и легче читается. К тому же, если мы решим изменить логин или пароль, нам будет достаточно поменять значение только в одном месте.

## Передача параметров через аргумент командной строки

Однако вы могли заметить, что у нас в тестовом сценарии остался еще один не очень приятный момент: мы вынуждены указывать полный путь к iso-образам: при объявлении виртуальной машины `my_ubuntu` в атрибуте `iso`, а также в действии `plug dvd` при установке гостевых дополнений.

Конечно, мы могли бы, как и в предыдущем пункте, объявить параметр `iso_dir "/opt/iso"`, однако в этом случае мы привязываем наш тестовый сценарий к конкретному местоположению необходимых iso-образов. Конечно, это не очень здорово.

В языке Testo-lang существует возможность указывать параметры не только в виде языковой конструкции, но и как аргумент командной строки. Давайте посмотрим на примере.

```testo
machine my_ubuntu {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server.iso"
}
...
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "${ISO_DIR}/testo-guest-additions.iso"
		...
	}
}
```

Мы изменили тестовый сценарий таким образом, что теперь папка с нужными iso-файлами указывается в качестве параметра `ISO_DIR` (заметьте, что обращаться к параметрам можно и в объявлении виртуальных машин). Однако мы нигде не объявляли параметр `ISO_DIR`. Если мы попытаемся запустить этот тестовый сценарий таким же способом, как в предыдущем уроке, то увидим ошибку

<Terminal height="150px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
	<span className="">Can't construct VmController for vm my_ubuntu: target iso file /ubuntu_server.iso doesn't exist<br/></span>
	<span className="">user$ </span>
</Terminal>

Т.к. параметр `ISO_DIR` нигде не был объявлен, то при попытке выяснить его значение в атрибуте `iso` виртуальной машины `my_ubuntu` будет получена пустая строка, в которой затем добавится постфикс `/ubuntu_server.iso`

Для того, чтобы объявить параметр `ISO_DIR` мы будем использовать аргумент командной строки.

<Terminal height="100px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation --param ISO_DIR /opt/iso<br/></span>
</Terminal>

Таким образом, если по каким-то причинам расположение iso-файлов изменилось (нарпимер, сценарий запускается на другом компьютере), достаточно изменить один аргумент командной строки при запуске сценария, а менять сам сценарий не будет никакой необходимости.

При этом, если вы проделывали шаги из предыдущего урока и уже запускали скрипт в том виде, в котором он был в конце предыдущей части, то теперь после нового запуска вы увидите такой вывод

<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 3<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

То есть ни один тест так и не был выполнен. Это произошло потому, что все тесты являются **закешированными**, и прогонять их повторно без причины нет никакого смысла. Подробнее мы рассмотрим механизм кеширования в следующем уроке, и там же подробно разберем, почему в этом конкретном случае все тесты остались закешированными, несмотря на то, что мы довольно сильно их изменили.

А сейчас, просто для того чтобы убедиться в работоспособности нашего тестового сценария уже с наличием параметров, запустим интерпретатор `testo` с аргументом `invalidate`, после чего все тесты должны прогнаться заново.

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo --param ISO_DIR /opt/iso --invalidate \*<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
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
	<span className="yellow ">No network interfaces detected </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Hostname: </span>
	<span className="blue ">for 30s with interval 1s in virtual machine </span>
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
	<span className="green bold"> PASSED in 0h:4m:24s<br/></span>
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
	<span className="yellow ">"dpkg -i /media/*.deb" </span>
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
	<span className="green bold"> PASSED in 0h:0m:14s<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo Hello world<br/></span>
	<span className=" ">Hello world<br/></span>
	<span className=" ">+ echo from bash<br/></span>
	<span className=" ">from bash<br/></span>
	<span className="blue ">[ 67%] Executing python3 command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">Hello from python3!<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">guest_additions_demo</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_demo</span>
	<span className="green bold"> PASSED in 0h:0m:3s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:4m:42s<br/></span>
	<span className="blue bold">UP-TO-DATE: 0<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Итоги

С помощью параметров можно делать тестовые сценарии гораздо более читаемыми и гибкими.

Во-первых, можно "именовать" часто встречаемые константы чтобы в них было проще ориентироваться и применять в нужных участках сценариев.

Во-вторых, с помощью передачи параметров через аргументы командной строки можно управлять прогоном тестов, не меняя при этом сам текст тестового сценария.

Итоговый скрипт можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/4)
