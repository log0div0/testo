# Часть 9. Макросы

## С чем Вы познакомитесь

В этой части вы познакомитесь с механизмом макросов в платформе Testo, а также научитесь работать с несколькими файлами .testo в рамках одного проекта.

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [восьмой части](8_flash).

## Вступление

В ходе предыдущих уроков наш тестовый сценарий `hello_world.testo` стал содержать в себе достаточно много кода, и при этом часть кода явно дублируется. Например, вы могли заметить, что тесты `server_install_ubuntu` и `client_install_ubuntu` различаются фактически тем, что в них фигурируют разные виртуальные машины и значением параметров для имени хоста и логина, и при этом все действия в этих тестах дублируются.

Конечно же, в таких условиях хочется немного сэкономить место и инкапсулировать какие-то похожие участки кода в какие-нибудь конструкции. В обычных языках программирования для этого можно использовать функции, процедуры и другие конструкции, а в платформе Testo для этого существуют макросы.

В языке testo-lang макрос означает поименованный набор **действий**. При этом вызов макроса также является **действием**. Благодаря макросу можно объединять похожие участки тестовых сценариев и делать их более читаемыми и компактными.

Макросы могут принимать аргументы, в том числе аргументы по умолчанию. Благодаря аргументам можно обобщать разные участки тестовых сценариев, даже если они несколько отличаются друг от друга.

И, конечно, в языке `testo-lang` предусмотрена возможно разносить участки кода по разным файлам. Делается это с помощью директивы `include` и мы сегодня с ней тоже познакомимся.

## С чего начать?

В нашем тестовом сценарии мы делаем много подготовительных действий для машин `client` и `server`: устанавливаем ОС Ubuntu Server, гостевые допонения, вытаскиваем сетевую карточку, "смотряющую" в Интернет. По большей части, мы видим много довольно одинаковых действий, которые мы, тем не менее, в предыдущих частях просто дублировали. Однако мы можем сделать наши подготовительные тесты гораздо компактнее, если прибегнем к помощи макросов.

Давайте начнем с установки ОС. Сейчас тест на установку ОС Ubuntu выглядит так:

```testo
test server_install_ubuntu {
	server {
		start
		wait "English"
		press Enter
		#Действия могут разделяться символом новой строки
		#или точкой с запятой
		wait "Install Ubuntu Server"; press Enter;
		wait "Choose the language";	press Enter
		wait "Select your location"; press Enter
		wait "Detect keyboard layout?";	press Enter
		wait "Country of origin for the keyboard"; press Enter
		wait "Keyboard layout"; press Enter
		#wait "No network interfaces detected" timeout 5m; press Enter
		wait "Primary network interface"; press Enter
		wait "Hostname:" timeout 5m; press Backspace*36; type "${server_hostname}"; press Enter
		wait "Full name for the new user"; type "${server_login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "${default_password}"; press Enter
		wait "Re-enter password to verify"; type "${default_password}"; press Enter
		wait "Use weak password?"; press Left, Enter
		wait "Encrypt your home directory?"; press Enter
		
		#wait "Select your timezone" timeout 2m; press Enter
		wait "Is this time zone correct?" timeout 2m; press Enter
		wait "Partitioning method"; press Enter
		wait "Select disk to partition"; press Enter
		wait "Write the changes to disks and configure LVM?"; press Left, Enter
		wait "Amount of volume group to use for guided partitioning"; press Enter
		wait "Write the changes to disks?"; press Left, Enter
		wait "HTTP proxy information" timeout 3m; press Enter
		wait "How do you want to manage upgrades" timeout 6m; press Enter
		wait "Choose software to install"; press Enter
		wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
		wait "Installation complete" timeout 1m; 

		unplug dvd; press Enter
		wait "${server_hostname} login:" timeout 2m; type "${server_login}"; press Enter
		wait "Password:"; type "${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

При этом для установки клиента мы видим практически такую же картину, за исключением того, что там фигурирует виртуальная машина `client` вместо `server` и используются другие `hostname` и `login`. Этот набор действий - отличный кандидат на внедрение макроса.

Давайте объявим [макрос](/docs/lang/macro) `install_ubuntu` (на том же уровне, что и объяевление тестов, сущностей и параметров).

```testo
macro install_ubuntu(hostname, login, password) {
	start
	wait "English"
	press Enter
	#Действия могут разделяться символом новой строки
	#или точкой с запятой
	wait "Install Ubuntu Server"; press Enter;
	wait "Choose the language";	press Enter
	wait "Select your location"; press Enter
	wait "Detect keyboard layout?";	press Enter
	wait "Country of origin for the keyboard"; press Enter
	wait "Keyboard layout"; press Enter
	#wait "No network interfaces detected" timeout 5m; press Enter
	wait "Primary network interface"; press Enter
	wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
	wait "Full name for the new user"; type "${login}"; press Enter
	wait "Username for your account"; press Enter
	wait "Choose a password for the new user"; type "${password}"; press Enter
	wait "Re-enter password to verify"; type "${password}"; press Enter
	wait "Use weak password?"; press Left, Enter
	wait "Encrypt your home directory?"; press Enter
	
	#wait "Select your timezone" timeout 2m; press Enter
	wait "Is this time zone correct?" timeout 2m; press Enter
	wait "Partitioning method"; press Enter
	wait "Select disk to partition"; press Enter
	wait "Write the changes to disks and configure LVM?"; press Left, Enter
	wait "Amount of volume group to use for guided partitioning"; press Enter
	wait "Write the changes to disks?"; press Left, Enter
	wait "HTTP proxy information" timeout 3m; press Enter
	wait "How do you want to manage upgrades" timeout 6m; press Enter
	wait "Choose software to install"; press Enter
	wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
	wait "Installation complete" timeout 1m; 

	unplug dvd; press Enter
	wait "${hostname} login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"
}
```

Вы можете заметить, что внутри макроса не фигурирует никаких виртуальных машин. В сущности, наш новый макрос представляет собой все действия по установке ОС, но без "прикрепления" к конкретной виртуальной машине.

Обратите внимание, что внутри макроса мы всё еще обращаемся к параметрам: `type "${hostname}"`, `type "${login}"` и так далее. Но теперь мы не имеем виду конкретное значение определённого параметра, а вместо этого обращаемся к значению аргументов макроса. Конкретное значение этих аргументов, конечно же, будет отличаться в разных вызовах макроса.

Давайте теперь попробуем преобразовать наши тесты `server_install_ubuntu` и `client_install_ubuntu`:

```testo
test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}", "${default_password}")
}

test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}", "${default_password}")
}
```

Гораздо компактнее, не правда ли? Обратите внимание, что вызов макроса работает точно так же, как и любое другое действие. Значение для аргументов макроса мы берем, обращаясь к соответствующим виртуальным машинам параметрам.

Давайте попробуем запустить наши тестовые сценарии.

<Terminal height="400px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 10<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Что же мы наблюдаем? Мы видим, что все наши тесты остались закешированными, несмотря на то, что мы очень сильно (как кажется) изменили текст самых базовых тестов. Однако платформа Testo при рассчёте контрольных сумм тестовых сценариев при встрече с вызовом макроса "раскрывает" его в полноценный набор действий и принимает во внимание только содержимое макроса. Т.к. содержимое тестов реально не изменилось (оно лишь перекочевало в макрос), то и контрольная сумма остаётся целостной, а значит кеш остаётся действительным.

Наши тесты стали гораздо компактнее, но, как это ни странно, их можно сделать ещё компактнее. Для этого давайте заметим, что в обоих тестах мы передаём в качестве пароля значение `${default_password}` - нас такое значение абсолютно устраивает в обоих виртуальных машинах. В этом случае мы можем сделать аргумент `password` в макросе `install_ubuntu` аргументом по умолчанию.

```testo
param default_password "1111"
macro install_ubuntu(hostname, login, password = "${default_password}") {
	start
	wait "English"
	press Enter
	...
```

Обратите внимание, значения аргумента по умолчанию мы высчитываем на основе значения параметра `default_password`. Чтобы такая схема правильно работала, параметр `default_password` должен быть объявлен до объявления макроса.

В самих тестах теперь можно не указывать аргумент `password` при вызове макроса:

```testo
test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}

test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}")
}
```

Если запустить тестовый сценарий сейчас, то мы увидим ровно такую же картину: все тесты закешированы. И снова причина кроется в том, что при рассчете контрольной суммы тестов принимается во внимание только итоговый вид тестов после "разворачивания" макросов.

<Terminal height="400px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 10<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Установка гостевых дополнений

Теперь давайте займёмся установкой гостевых дополнений. Конечно же, и для клиента и для сервера установка гостевых дополнений выглядит практически одинаково, поэтому здесь мы можем поступить так же, как и с установкой ОС

```testo
param guest_additions_pkg "testo-guest-additions*"

macro install_guest_additions(hostname, login, password="${default_password}") {
	plug dvd "${ISO_DIR}/testo-guest-additions.iso"

	type "sudo su"; press Enter;
	#Обратите внимание, обращаться к параметрам можно в любом участке строки
	wait "password for ${login}"; type "${password}"; press Enter
	wait "root@${hostname}"

	type "mount /dev/cdrom /media"; press Enter
	wait "mounting read-only"; type "dpkg -i /media/${guest_additions_pkg}"; press Enter;
	wait "Setting up testo-guest-additions"
	type "umount /media"; press Enter;
	#Дадим немного времени для команды umount
	sleep 2s
	unplug dvd
}
```

Обратите внимание, что внутри этого макроса мы обращаемся к параметрам `ISO_DIR` и `guest_additions_pkg`, несмотря на то, что они не входят в список аргументов. Такая схема успешно работает благодаря алгоритму разрешения имён параметров при обращении к ним с помощью оператора `${}`:

1. Если обращение к параметру происходит внутри макроса, то сначала проверяется, входит ли этот параметр в список аргументов макроса. Если входит, то поиск значения параметра на этом прекращается. Например, в нашем макросе алгоритм завершит работу на этом шаге при обращении к `${hostname}`, `${login}` и `${password}`.
2. Происходит поиск значений глобально объявленных параметров (в том числе параметров, объявленных с помощью аргумента `--param`). Если нужное значение найдено, то поиск на этом завершается. В нашем макросе алгоритм завершит работу на этом шаге при обращении к `${ISO_DIR}` и `${guest_additions_pkg}`.
3. Если на предыдущих шагах ниичего не было найдено, то возвращается пустое значение.

Сами тесты при использовании макроса становятся гораздо компактнее

```testo
test server_install_guest_additions: server_install_ubuntu {
	server install_guest_additions("${server_hostname}", "${server_login}")
}

test client_install_guest_additions: client_install_ubuntu {
	client install_guest_additions("${client_hostname}", "${client_login}")
}
```

И вновь, если запустить тесты, мы увидим, что всё закешировано:

<Terminal height="400px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 10<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Напоследок мы предлагаем вам самостоятельно вынести в макрос действия из теста `client_unplug_nat`.

## Директива include

Конечно, благодаря макросам наш файл `hello_world.testo` стал гораздо компактнее, но в нем все еще есть ощущение "кучи малы". В одном файле у нас расположены и объявления сущностей, и подготвительные тесты, и "боевые" тесты. Сейчас это может не доставлять особых неудобств, но в будущем, с ростом количества кода, разложение кода по полочкам будет все более и более актуальным. Давайте попробуем реорганизовать наши файлы и наш код внутри них.

Вместо одного файла `hello_world.testo` у нас появится несколько файлов: `declarations.testo`, `macros.testo` и `tests.testo`. В файл `declarations.testo` мы занесем все объявления виртуальных сущностей (`machine`, `flash` и `network`), а также параметры; в файл `macros.testo` перенесём все макросы, и все тесты будут лежать в `tests.testo`. Конечно, надо понимать, что такое разделение достаточно условное и что вы можете перемещать код из разных файлов так, как вам удобнее.

Конечно же, сам по себе набор файлов не даёт возможности рассматривать их как части одного тестового проекта. Для этого необходимо эти файлы связать между собой. В языке `testo-lang` для этого используется знакомый многим механизм включения файлов `include`.

В нашем проекте файл `declarations.testo` не зависит ни от чего, поэтому он не нуждается в директиве `include`. Файл `macros.testo` же зависит от `declarations.testo`, потому что для макросы используют параметры `default_password` и `guest_additions_pkg`, которые объявлены в `declarations.testo`. Для правильного функционирования подсчета контрольных сумм нам необходимо удостовериться, что эти параметры точно будут объявлены на момент объявления макросов. Поэтому в начале файла `macros.testo` необходимо добавить директиву `include`

```testo
include "declarations.testo"

macro install_ubuntu(hostname, login, password = "${default_password}") {
	...
```

Файл с тестами `tests.testo` явно зависит как от `declarations.testo`, так и от `macros.testo`. Но т.к. `declarations.testo` уже включены в `macros.testo`, нам достаточно включить только `macros.testo` 

```testo
include "macros.testo"

test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}
...
```
Теперь все наши  тестовые сценарии выглядят достаточно компактно и расположены по полочкам. Остаётся вопрос, как же теперь запускать наши тесты? Для этого есть два способа:

1. Указание "конечного" файла с тестами: `sudo testo run tests.testo --stop_on_fail --param ISO_DIR /opt/iso`
2. Указание целой папки с тестами: `sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso`

В любом случае вы должны снова увидеть, что все тесты закешированы, потому что реально тествые сценарии мы так и не поменяли за весь текущий урок:

<Terminal height="400px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 10<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Итоги

Макросы и связывание файлов с помощью директивы `include` позволяют существенно упростить и реорганизовать код тестовых сценариев, сделать его гораздо более читаемым и понятным. Чем больше у вас будет кода, тем больше и больше вам будет нужна инкапсуляция и разнесение кода по файлам. Постарайтесь начать этот процесс как можно раньше, чтобы не превращать свои тестовые сценарии в одную большую сплошную "кучу малу". Механизм кеширования в Testo позволяет вам не перезапускать уже успешно пройденные тесты, даже после внедрения макросов, если все сделать правильно и аккуратно.

Готовые скрипты можно найти [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/9)