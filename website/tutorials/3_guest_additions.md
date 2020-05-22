# Часть 3. Гостевые дополнения

## С чем Вы познакомитесь

В этой части вы познакомитесь с иерархической организацией тестов, а также с процессом установки и использования гостевых дополнений на примере ОС Ubuntu Server 16.04

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
4. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
5. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
6. (Рекомендовано) Проделаны шаги из [второй части](2_ubuntu_installation).

## Вступление

В прошлой части мы научились автоматизировать установку Ubuntu Server 16.04 и познакомились с основными действиями в языке Testo-lang, которые позволяют имитировать действия человека. Такой подход хорош с двух точек зрения:
1. Позволяет тестировать систему ровно в том же виде и ровно таким же способом, как это будет делать конечный пользователь - человек
2. Не требует наличия в виртуальной машине никаких специальных агентов

Однако имитация действий человека обладает и одним недостатком: неудобное выполнение консольных команд и других вспомогательных действий. В самом деле. для того, чтобы выполнить bash-команду, имитируя действия человека, необходимо написать что-то вроде

```testo
type "command_to_execute";
press Enter
type "echo Result is $?"; press Enter
wait "Result is $?"
```

При этом, конечно, было бы удобно выполнить какое-то одно действие вроде

```testo
exec bash "command_to_execute"
```

И положиться на код возврата этой команды

Платформа Testo помогает решить эту проблему и предоставляет гостевые дополнения для виртуальных машин с разными ОС, в том числе Linux-дистрибутивах Ubuntu и CentOS, а также Windows 7  и Windows 10. Наличие установленных гостевых дополнений на виртуальной машине открывает возможность использовать новые действия: [`exec`](/docs/lang/actions#exec) - выполнение скриптов на разнах языках, [`copyto`](/docs/lang/actions#copyto) - копирование файлов внутрь виртуальной машины с хоста и [`copyfrom`](/docs/lang/actions#copyfrom) - Копирование файлов из виртуальной машины на хост.

Гостевые дополнения рекомендуется устанавливать на виртуальные машины в двух случаях:
1. Если виртуальная машина является вспомогательной и ее целостность с точки зрения неважна (например, на генераторе трафика при тестировании DPI-систем)
2. Если наличие гостевых дополнений никак не влияет на поведение тестируемого ПО

Если суммировать вышеизложенное, то можно сделать вывод, что гостевые дополнения следует по возможности устанавливать на виртуальные машины, и лишь в случае когда это невозможно или нежелательно от них стоит отказаться.

## С чего начать?

Для того, чтобы установить гостевые дополнения на уже установленную ОС, необходимо проделать несколько шагов внутри тестовых сценариев:
1. Подключить iso-образ с дополнениями в виртуальный dvd-привод виртуальной машины
2. Смонтировать подключенный dvd-привод в файловую систему ОС (при необходимости, если этого не происходит автоматически)
3. Выполнить установку гостевых дополнений в системе (способ зависит от целевой ОС)

Давайте вернемся к тому тестовому сценарию, который у нас получился в конце прошлой части, где мы автоматизировали установку Ubuntu Server 16.04

Начнем с того, что переименуем тест `my_first_test` во что-то более осмысленное. Например, в `ubuntu_installation`

```testo
test ubuntu_installation {
	my_ubuntu {
		start
		wait "English"
		...
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```


Пока не следует запускать новый сценарий, мы сделаем это немного позже. Давайте лучше приступим к установке гостевых дополений на нашу машину `my_ubuntu`.

Для этого необходимо познакомиться с понятием [иерархии тестов](/docs/getting_started/test_policy). Если вкратце - то все тесты выстраиваются по принципу "от простого к сложному", при этом более сложный тест зависит от результата работы более простого теста. Самые простые тесты, которые не зависят ни от кого, называются **базовыми**. В нашем примере тест `ubuntu_installation` как раз является базовым тестом.

Помимо базовых тестов существуют и **производные** тесты, которые запускаются только когда будут выполнены все родительские тесты. Для установки гостевых дополнений нам потребуется написать наш первый производный тест. Связь между тестами задается так:

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		abort "stop here"
	}
}
```

Давайте убедимся, что наш производный тест действительно зависит от базового.

Для этого давайте запустим прогон тестового сценария, но при этом используем новый аргумент командной строки `--test_spec`. Этот аргумент позволяет указывать, какой именно тест мы хотим выполнить (вместо того, чтобы выполнять все тесты подряд). Запустив выполнение, мы увидим в начале вывода следующую информацию:

<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
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

Этот аргумент позволяет указывать, какой именно тест мы хотим выполнить (вместо того, чтобы выполнять все тсеты подряд). Запустив выполнение, мы увидим в начале вывода следующую информацию:

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
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
	<span className="green bold">[ 50%] Test </span>
	<span className="yellow bold">ubuntu_installation</span>
	<span className="green bold"> PASSED in 0h:4m:22s<br/></span>
	<span className="blue ">[ 50%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Running test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:52:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_installation</span>
	<span className="red bold"> FAILED in 0h:0m:0s<br/></span>
	<span className="">user$ </span>
</Terminal>

 В начале вывода мы видим, что платорма Testo запланировала прогон теста `ubuntu_installation`, несмотря на то, что мы попросили запустить только guest_additions_installation. Это происходит потому, что мы пытаемся прогнать производный тест, при том что **базовый** тест еще не был проведен успешно. Поэтому платформа Testo автоматически попытается прогнать сначала базовый тест, а только затем перейдет к производному.

Но разве мы не устанавливали уже успешно Ubuntu? В конце прошлой части мы закончили на том, что Ubuntu Server уже была успешно установлена, тест закончился, и состояние должно было зафиксироваться.

Однако на самом деле тогда наш тест назывался `my_first_test`, и после переименования его в `ubuntu_installation` он выглядит для платформы Testo как совершенно новый тест, который никогда до этого не прогонялся.

В конце вывода мы увидим, что базовый тест был успешно выполнен, Testo приступило к прогону второго теста, но он закончился с ошибкой (из-за действия `abort`).

Если мы еще раз запустим Testo, то увидим уже следующее

<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Restoring snapshot </span>
	<span className="yellow ">ubuntu_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Running test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:52:3: Caught abort action on virtual machine my_ubuntu with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_installation</span>
	<span className="red bold"> FAILED in 0h:0m:0s<br/></span>
	<span className="">user$ </span>
</Terminal>

Это означает, что платформа Testo распознала, что `ubuntu_installation` уже был успешно выполнен и его состояние уже было зафиксировано. Поэтому вместо того, чтобы прогонять этот тест заново, можно просто восстановить стенд в то состояние, в котором он был зафиксирован на момент окончания `ubuntu_installation`.

## Устанавливаем дополнения

Пора переходить к установке самих дополнений. В прошлой части мы познакомились с действием `unplug dvd`, которое извлекает текущий подключенный iso-образ из виртуального dvd-привода. Конечно, есть действие `plug dvd` и для обратного действия, для подключения iso-образа. Это действие, в отличие от `unplug dvd`, принимает аргумент - путь к iso-образу, который необходимо подключить.

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"
		abort "stop here"
	}
}
```

Попробуйте запустить такой сценарий (не забудьте аргумент `--stop_on_fail`), дождитесь точки останова и с помощью virt-manager зайдите в свойства виртуальной машины. В разделе, посвященной cdrom, вы увидите информацию о подключенном iso-образе

![CDROM plugged](/static/tutorials/3_guest_additions/plugged_cdrom.png)

Теперь нам необходимо смонтировать подключенный dvd-привод в файловую систему Ubuntu. Т.к. для этого требуются root-права, то можно для начала войти в sudo-режим

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		abort "stop here"
	}
}
```

Ну а теперь переходим к установке самих дополнений. В конце не забудем отмонтировать cdrom и вытащить iso-образ из виртуального dvd-привода

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"; type "dpkg -i /media/*.deb"; press Enter;
		wait "Setting up testo-guest-additions"
		type "umount /media"; press Enter;
		#Дадим немного времени для команды umount
		sleep 2s
		unplug dvd
	}
}
```

Обратите внимание, что в сценарии появилось новое действие [`sleep`](/docs/lang/actions#sleep), которое работает ровно так, как это и можно представить: просто запускает безусловное ожидание на определенное количество времени.

В конце можно убрать `abort` и зафиксировать тест.

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_installation<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Restoring snapshot </span>
	<span className="yellow ">ubuntu_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Running test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 50%] Plugging dvd </span>
	<span className="yellow ">/opt/iso/testo-guest-additions.iso </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"sudo su" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">password for my-ubuntu-login </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">root@my-ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"mount /dev/cdrom /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">mounting read-only </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"dpkg -i /media/*.deb" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">Setting up testo-guest-additions </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"umount /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Sleeping in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> for 2s<br/></span>
	<span className="blue ">[ 50%] Unplugging dvd </span>
	<span className="yellow "> </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 50%] Taking snapshot </span>
	<span className="yellow ">guest_additions_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_installation</span>
	<span className="green bold"> PASSED in 0h:0m:17s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 2 TESTS IN 0h:0m:17s<br/></span>
	<span className="blue bold">UP-TO-DATE: 1<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

На этом установка гостевых дополнений закончена, давайте их испробуем в деле

## Пробуем гостевые дополнения в деле

Для тестирования гостевых дополнений сделаем новый производный тест, который теперь будет зависеть от `guest_additions_installation`. После установки гостевых дополнений у нас появляется в арсенале несколько новых действий. В этой части мы сосредоточимся на действии `exec`. Попробуем выполнить баш-скрипт, который выводит на экран "Hello world!" (Можно обойтись без точки останова `abort`)

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		exec bash "echo Hello world"
	}
}
```

Результат будет таким

<Terminal height="500px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">guest_additions_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo Hello world<br/></span>
	<span className=" ">Hello world<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">guest_additions_demo</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_demo</span>
	<span className="green bold"> PASSED in 0h:0m:4s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:4s<br/></span>
	<span className="blue bold">UP-TO-DATE: 2<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Видно, как была выполнена баш-команда. Вообще, `exec` не ограничивается запуском баш-команд. Например, можно запускать питоновские скрипты (конечно, если в гостевой системе вообще установлен интерпретатор python). При этом скрипты могут быть и многострочными, для этого их нужно заключать в тройные кавычки

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		exec bash """
			echo Hello world
			echo from bash
		"""
		#Двойные кавычки внутри скриптов необходимо экранировать
		exec python2 "print('Hello from python2!')"
		exec python3 "print('Hello from python3!')"
	}
}
```

<Terminal height="650px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --test_spec guest_additions_demo<br/></span>
	<span className="">Some tests have lost their cache:<br/></span>
	<span className="">	- guest_additions_demo<br/></span>
	<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">guest_additions_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
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
	<span className="green bold"> PASSED in 0h:0m:5s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:5s<br/></span>
	<span className="blue bold">UP-TO-DATE: 2<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

С остальными возможностями гостевых дополнений мы познакомимся в следующих уроках. В частности, действие `copyto` разбирается в [пятой](5_caching) части уроков

## Итоги

В результате этого урока у нас получилось следующее дерево тестов

![](/static/tutorials/3_guest_additions/tests_tree.png)

Итоговый скрипт можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/3)