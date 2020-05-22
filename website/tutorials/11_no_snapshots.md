# Часть 11. No snapshots

## С чем Вы познакомитесь

В этой части вы познакомитесь с тестами без снепшотов гипервизора в платформе Testo, которые могут позволить сэкономить вам очень много места на диске.

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [десятой части](10_if).

## Вступление

Как, наверное, вы уже прочуствовали, в платформе Testo большое значение имеет кешируемость тестов, что позволяет не прогонять с нуля каждый раз все тесты, а отталкиваться от уже успешно пройденных (конечно, если успешно пройденные тесты имеют актуальный кеш). В первую очередь, такое поведение становится возможным благодаря возможности создавать снепшоты виртуальных машин и флешек, а затем возвращаться к сохраненным состояниям при необходимости прогнать производные тесты.

Но такое поведение рано или поздно может привести к проблемам: каждый снепшот (как машин, так и флешек) занимает немало места на диске, и это самое место может очень быстро закончиться. Ситуация может усугубиться тем, что в конце успешного теста создаются снепшоты **всех** сущностей, участвующих в тесте. Если, например, тест включает в себя пять виртуальных машин, то для каждой из них будет создан снепшот в конце теста.

Поэтому в языке `testo-lang` существует механизм, который позволяет не создавать "твёрдые" снепшоты виртуальных сущностей, а ограничиваться фиксацией тестов в виде метаданных. Грамотное использование этого механизма позволит вам сэкономить очень много места на диске без особого ущерба, и именно с этим механизмом мы познакомимся в этом уроке.

## С чего начать?

Давайте посмотрим на наше дерево тестов, которое у нас накопилось на текущий момент.

![Tests hierarchy](/static/tutorials/8_flash/test_hierarchy.png)

Всего у нас имеется 8 тестов, и в конце каждого теста сейчас создаются снепшоты, которые уже занимают довольно места на нашем диске. Допустим, мы хотели бы немного поправить эту ситуацию.

Давайте подумаем, а зачем вообще нужны снепшоты в конце каждого успешного теста? В первую очередь это необходимо для того, чтобы платформа Testo могла бы вернуться к сохранённым снепшотам в случае, если нужно прогнать производные тесты. например, если бы в нашем дереве тестов тест `test_ping` потерял бы целостность, то нам необходимы были бы снепшоты `server_prepare` и `client_prepare` для того, чтобы заново прогнать этот тест.

Но зачем нужны снепшоты в конце тестов `test_ping` и `exchange_files_with_flash`? Ведь эти тесты являются конечными в нашей иерархии, и возвращаться к состоянию виртуальных машин на момент их успешного окончания просто бесмыссленно на данный момент. Поэтому давайте попробуем убрать в этих двух тестах снепшоты:

```testo
[no_snapshots: true]
test test_ping: client_prepare, server_prepare {
	client exec bash "ping 192.168.1.2 -c5"
	server exec bash "ping 192.168.1.1 -c5"
}

[no_snapshots: true]
test exchange_files_with_flash: client_prepare, server_prepare {
	client {
		#Создаём файл, который нужно будет передать на сервер
		exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
		...
```

Здесь мы воспользовались ещё одной новой для нас языковой конструкцией - атрибутами [тестов](/docs/lang/test). Всего существует два возможных атрибута тестов: `no_snapshots` и `description`. Атрибут `description` мы разберем в будущем уроке, а сейчас нам достаточно определить атрибут `no_snapshots` и задать ему значение `true`.

Попробуем запутстить тесты (на текущий момент все тесты должны быть проведены и закешированы).

<Terminal height="620px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 80%] Preparing the environment for test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 80%] Running test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
	<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.052 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.034 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.037 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.038 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.045 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3996ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.034/0.041/0.052/0.007 ms<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
	<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.055 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.037 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.046 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.039 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.037/0.044/0.055/0.009 ms<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[ 90%] Test </span>
	<span className="yellow bold">test_ping</span>
	<span className="green bold"> PASSED in 0h:0m:12s<br/></span>
	<span className="blue ">[ 90%] Preparing the environment for test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Running test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo 'Hello from client!'<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
	<span className=" ">Hello from client!<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">exchange_files_with_flash</span>
	<span className="green bold"> PASSED in 0h:0m:16s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:28s<br/></span>
	<span className="blue bold">UP-TO-DATE: 8<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Мы видим, что оба наших изменённых теста потеряли целостность и их пришлось прогонять заново. Это произошло потому, что атрибуты теста также входят в подсчёт целостности кеша.

Но что же теперь? Теперь в конце наших тестов не создаются снепшоты, и можно было бы подумать, что теперь эти тесты никогда не будут кешироваться, и они будут прогняться каждый раз, так? На самом деле, это совсем не так! Давайте запустим наши тесты ещё раз.

<Terminal height="350px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
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

Что же мы видим? Все тесты закешированы, и ничего не пришлось прогонять. И при этом два последних теста имеют атрибут `no_snapshots`, то есть у виртуальных машин и флешек не создаются снепшоты (в этом кстати можно легко убедиться, посмотрев свойства виртуальных машин):

![No snapshots](/static/tutorials/11_no_snapshots/no_snapshots.png)

Почему же так происходит? Давайте разбираться.

Дело в том, что в платформе Testo существуют два типа снепшотов, которые работают одновременно никак не мешают друг другу:

1. Снепшоты на уровне метаданных. Это различные служебные данные в виде небольших текстовых файликов, которые создаёт Testo для своих целей и которые участвуют, в том числе, в контроли целостности кеша тестов. Эти снепшоты создаются всегда и вы не можете на них повлиять. Если вы приглядитесь на вывод последнего прогона тестов, вы моожете увидеть, что в выводе всё-равно присутствют надписи `Taking snanshot...` - это как раз и отражает создание новых метаданных.
2. Снепшоты на уровне сущностей. Это и есть снепшоты в привычном нам понимании: снепшоты виртуальных машин в гипервизоре или реальные копии флешек, к которым можно откатиться при необходимости. Эти снепшоты создаются только в том случае, если у теста нет атрибута `no_snapshots`. Т.к. сейчас мы включили этот атрибут, то снепшоты на уровне сущностей на были созданы.

Если кратко просуммировать вышесказанное, то получится очень важное заключение

> Атрибут `no_snapshots` не влияет на кешируемость тестов, и к таким тестам применяются точно такие же законы целостности кеша, как и к обычным тестам. Наличие такого атрибута **не означает**, что тест не будет кешироваться и будет всегда запускаться.

Получается, что мы сэкономили немного места на диске и при этом совершенно ничего не потеряли - ведь снепшоты тестов `test_ping` и `exchange_files_with_flash` нам и так никогда бы не пригодились. Отсюда следует ещё одно важное заключение

> Для "листовых" тестов (тестов, не имеющих потомков) можно и даже рекомендуется указывать атрибут `no_snapshots`, т.к. практической пользы от наличия снепшотов в таких тестах всё равно практически нет.

## no_snapshots в промежуточных тестах

Может возникнуть ощущение, что если атрибут `no_snapshots` экономит место на диске и не влияет на кешируемость тестов, то может быть имеет смысл вообще указывать этот атрибут во всех тестах подряд? Такое ощущение будет не совсем верным.

Да, с одной стороны такой атрибут не влияет на кешируемость тестов, но это не значит, что его наличие не будет иметь никаких отрицательных эффектов. Давайте продемонстрируем это на примере и добавим этот атрибут в тест `client_unplug_nat`:

```testo
[no_snapshots: true]
test client_unplug_nat: client_install_guest_additions {
	client unplug_nat("${client_hostname}", "${client_login}", "1111")
}
```

Давайте прогоним именно этот тест и ничего больше.

<Terminal height="600px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">client_install_guest_additions</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Calling macro </span>
	<span className="yellow ">unplug_nat(</span>
	<span className="yellow ">hostname="client"</span>
	<span className="yellow ">, </span>
	<span className="yellow ">login="client-login"</span>
	<span className="yellow ">, </span>
	<span className="yellow ">password="1111"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Shutting down virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 1m<br/></span>
	<span className="blue ">[ 67%] Unplugging nic </span>
	<span className="yellow ">nat </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Starting virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">client login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"client-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">client_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">client_unplug_nat</span>
	<span className="green bold"> PASSED in 0h:0m:27s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:27s<br/></span>
	<span className="blue bold">UP-TO-DATE: 2<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Убедимся также, что этот тест закеширован, несмотря на `no_snapshots: true`

<Terminal height="250px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_unplug_nat<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:0s<br/></span>
	<span className="blue bold">UP-TO-DATE: 3<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 0<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

А теперь попытаемся прогнать тест `client_prepare`, который зависит от `client_unplug_nat`

<Terminal height="600px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes --test_spec client_prepare<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="blue ">[ 60%] Preparing the environment for test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 60%] Restoring snapshot </span>
	<span className="yellow ">client_install_guest_additions</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Running test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 60%] Calling macro </span>
	<span className="yellow ">unplug_nat(</span>
	<span className="yellow ">hostname="client"</span>
	<span className="yellow ">, </span>
	<span className="yellow ">login="client-login"</span>
	<span className="yellow ">, </span>
	<span className="yellow ">password="1111"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Shutting down virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 1m<br/></span>
	<span className="blue ">[ 60%] Unplugging nic </span>
	<span className="yellow ">nat </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Starting virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Waiting </span>
	<span className="yellow ">client login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Typing </span>
	<span className="yellow ">"client-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 60%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[ 80%] Test </span>
	<span className="yellow bold">client_unplug_nat</span>
	<span className="green bold"> PASSED in 0h:0m:29s<br/></span>
	<span className="blue ">[ 80%] Preparing the environment for test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Running test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 80%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/rename_net.sh /opt/rename_net.sh<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 80%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ chmod +x /opt/rename_net.sh<br/></span>
	<span className=" ">+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/></span>
	<span className=" ">Renaming success<br/></span>
	<span className=" ">+ ip a a 192.168.1.2/24 dev server_side<br/></span>
	<span className=" ">+ ip l s server_side up<br/></span>
	<span className=" ">+ ip ad<br/></span>
	<span className=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
	<span className=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
	<span className=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 ::1/128 scope host <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">2: server_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
	<span className=" ">    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/></span>
	<span className=" ">    inet 192.168.1.2/24 scope global server_side<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 fe80::5054:ff:fe00:aa/64 scope link tentative <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">client_prepare</span>
	<span className="green bold"> PASSED in 0h:0m:8s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:37s<br/></span>
	<span className="blue bold">UP-TO-DATE: 3<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

И мы с вами увидим очень интересную картину: тест `client_unplug_nat` одновременно помечен и как `UP-TO-DATE` и как `TEST TO RUN`. Давайте разберёмся, почему так происходит.

Когда Testo сканирует дерево тестов чтобы определить, какие тесты необходимо выполнить, каждый тест анализируется отдельно. Т.к. мы хотим запустить тест `client_prepare`, то предварительно анализируются на предмет закешированности все его предки: `client_install_ubuntu`, `client_install_guest_additions` и `client_unplug_nat`. Все эти тесты имеют валидный кеш, поэтому они и помечаются как `UP-TO-DATE`, что мы и видим.

Затем приходит черед проанализировать закешированность самого теста `client_prepare`. Кеш этого теста недействителен (потому что мы ранее меняли в предыдущем шаге `client_unplug_nat`), а значит его необходимо выполнить. Остаётся вопрос - а как его выполнить?

Если бы тест `client_unplug_nat` не был бы помечен, как `no_snapshots`, то мы могли бы откатиться к снепшотам виртуальных машин на момент окончания `client_unplug_nat`. Но у этого теста нет снепшотов, поэтому и откатиться нам некуда. Возникает вопрос - а как вернуть виртуальную машину `client` в состояние на момент окончания `client_unplug_nat`? В этом случае платформа Testo начинает идти вверх по иерархии тестов в попытке найти хоть один снепшот, за которой можно было бы "зацепиться". В нашем случае - это тест `client_install_guest_additions` - потому что он не был помечен атрибутом `no_snapshots`.

В итоге мы откатываемся к снепшоту `client_install_guest_additions` и начинаем заново прогонять тест `client_unplug_nat` **чтобы вернуть виртуальную машину `client` в нужное состояние** (в состояние на момент окончания теста `client_unplug_nat`). Именно поэтому мы видим `client_unplug_nat` в списке `TESTS TO RUN`.

После того, как мы вернули машину в нужное состояние, мы можем, наконец, запустить сам тест `client_prepare`. Этот процесс можно визуализировать примерно так

![Tests resolve](/static/tutorials/11_no_snapshots/search.png)

Если бы тест `client_install_guest_additions` был бы тоже помечен как `no_snapshots`, то итоговый план выполнения тестов выглядел бы так: `client_install_guest_additions->client_unplug_nat->client_prepare`

Попробуйте сбросить кеш теста `client_unplug_nat` (поменяйте в нем что-нибудь или используйте `invalidate`) и убедитесь, что тест `client_unplug_nat` также выполняется.

После этого давайте попробуем запустить все тесты целиком


<Terminal height="600px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 80%] Preparing the environment for test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 80%] Running test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
	<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.071 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.036 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.041 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.038 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.046 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3996ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.036/0.046/0.071/0.014 ms<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
	<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.060 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.036 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.066 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.043 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.065 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.036/0.054/0.066/0.012 ms<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[ 90%] Test </span>
	<span className="yellow bold">test_ping</span>
	<span className="green bold"> PASSED in 0h:0m:13s<br/></span>
	<span className="blue ">[ 90%] Preparing the environment for test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Running test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo 'Hello from client!'<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
	<span className=" ">Hello from client!<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">exchange_files_with_flash</span>
	<span className="green bold"> PASSED in 0h:0m:15s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:29s<br/></span>
	<span className="blue bold">UP-TO-DATE: 8<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Что же мы видим? Мы видим, что несмотря на то, что наш тест `client_unplug_nat` заимел атрибут `no_snapshots`, наше время прогона **ЛИСТОВЫХ** тестов не пострадало: ведь мы всё еще можем использовать снепшоты от теста `client_prepare`.

> Получается, что применение атрибута `no_snapshots` в промежуточных тестах может сэкономить место на диске, но в некоторых случаях в ущерб времени прогона тестов.

Попробуйте самостоятельно добавить атрибут `no_snapshots` в тест `server_unplug_nat` и внимательно поизучайте какие тесты в каких случаях будут запускаться.

Теперь давайте рассмотрим ещё один момент, после которого мы сформулируем несколько правил относительно того, где стоит применять `no_snapshots`, а где не стоит.

## no_snapshots в опорных тестах - плохая идея

Перед дальнейшим шагом убедитесь, что тесты `client_unplug_nic`, `server_unplug_nic`, `test_ping` и `exchange_files_with_flash` имеют атрибут `no_snapshots`, все тесты должны быть прогнаны и закешированы.

При таком раскладе получается, что мы сэкономили довольно много места и при условии что мы не будем трогать тесты `client_prepare` и `server_prepare` тесты `test_ping` и `exchange_files_with_flash` прогоняются так же быстро, как если бы все тесты имели полноценные снепшоты. Мы достигли определённого баланса - места гораздо меньше, и неудобства от увеличенного времени прогона тестов пока не сильно ощущаются.

Но давайте продемонстрируем, что иногда очень важно бывает вовремя остановиться в попытках сэкономить используемое место.

Давайте добавим `no_snapshots` в тесты `client_prepare` и `server_prepare` и попробуем прогнать все тесты


<Terminal height="350px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec test_ping --assume_yes<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	...
	<span className="">user$ </span>
</Terminal>

Посмотрите насколько выросла очередь `TESTS TO RUN`. Мы видим, что тесты `server_unplug_nat`, `client_unplug_nat`, `server_prepare` и `client_prepare` планируются к выполнению аж два раза! Давайте проанализируем, почему так происходит:

1. Нам необходимо выполнить два листовых теста: `test_ping` и `exchange_files_with_flash`, которые оба полагаются на опорные тесты `client_prepare` и `server_prepare`;
2. Т.к. тесты `client_prepare` и `server_prepare` теперь не имеют реальных снепшотов, платформе Testo ничего не остаётся кроме как выполнить поиск ближайших тестов, к результатам которых можно откатиться;
3. Сначала такой поиск происходит для теста `test_ping`, в результате появляется путь выполнения `server_unplug_nat->server_prepare->client_unplug_nat->client_prepare`, который берет свое начало с момента успешного окончания тестов `client_install_guest_additions` и `server_install_guest_additions`;
4. Такой же поиск приходится проделать и для теста `exchange_files_with_flash`! Ведь отсутствие снепшотов никто не отменял. В итоге некоторые тесты запланированы к выполнению аж два раза!

В данном случае мы, конечно, ещё сэкономили место на диске, но с другой стороны явно перегнули палку с точки зрения времени выполнения тестов. Наше решение никак нельзя назвать эффективным: неудобства значительно превысили пользу.

В связи с этим возникает вопрос, а какое же распредение атрибутов `no_snapshots` в тестах можно назвать оптимальным (в большинстве случаев)? Мы предлагаем такой алгоритм:

1. Все листовые (не имеющие потомков) тесты можно безболезненно помечать атрибутом `no_snapshots`, поэтому следует это сделать;
2. Промежуточные тесты следует помечать атрибутом `no_snapshots`, если эти тесты не являются **опорными**, то есть если к результатам их выполнения не будет происходить откатов (по крайней мере часто);
3. Тесты, имеющие несколько потомков, **не следует** помечать атрибутом `no_snapshots`.

Если попытаться применить этот алгоритм к нашему дереву тестов, то получается следующая картина:

1. Тесты `test_ping` и `exchange_files_flash` являются листовыми, поэтому им следует назначить атрибут `no_snapshots`;
2. Тесты `client_prepare` и `server_prepare` явно **не должны** иметь атрибут `no_snapshots`, т.к. от этих тестов зависит более одного теста;
3. Тесты `client_unplug_nat` и `server_unplug_nat` следует пометить атрибутом `no_snapshots` в том случае, если тесты `client_prepare` и `server_prepare` будут закешированы большую часть времени. Если эти тесты будут постоянно терять целостность кеша, то лучше оставить тесты `client_unplug_nat` и `server_unplug_nat` в первоначальном виде;
4. Тесты `install_ubuntu` достаточно трудоемкие и выполняются долго. Их можно оставить в изначальном виде, даже с учётом, что к ним редко будет происходить откат, просто чтобы сэкономить себе время на лишней установке ОС в том случае в случае непредвиденных обстоятельств;
5. Тесты `install_guest_additions` можно пометить атрибутом `no_snapshots`.

Проведя такую оптимизацию, мы достигнем неплохого баланса между экономией места на диске и скоростью прогона тестов. Многие подготовительные тесты имеют атрибут `no_snapshots`, потому что мы предполагаем, что подготовка будет происходить редко (в идеале - ровно один раз), после чего мы фиксируем итоги подготовки в тестах `client_prepare` и `server_prepare`. Производные сложны тесты, котрые предположительно будут прогоняться часто, всегда смогут положиться на результат тестов `prepare`, поэтому время их прогона не пострадает.

Этот алгоритм не является панацеей и универсальным решением. Конечно, в разных случаях могут быть свои нюансы, которые потребуют принимать во внимание другие факторы. Не бойтесь экспериментировать!

## Итоги

Механизм `no_snapshots` в языке `testo-lang` позволяет экономить место на диске в ущерб времени прогона тестов. Впрочем, если грамотно распоряжаться этим механизмом, то потеря времени на прогонах тестов может оказаться очень несущественной или вовсе отсутствовать. Поэтому при достижении определенного количества тестов определенно следует остановиться и подумать, какие тесты будут выполняться часто, а какие редко - и в каких тестах можно безболезненно поставить атрибут `no_snapshots`.

Готовые скрипты можно найти [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/11)