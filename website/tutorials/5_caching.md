# Часть 5. Кеширование

## С чем Вы познакомитесь

В этой части вы познакомитесь с механизмом кеширования в платформе Testo.

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
4. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
5. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
6. (Рекомендовано) Проделаны шаги из [четвертой части](4_params).

## Вступление

Платформа Testo построена на принципе "Если вы хотит что-то сделать с виртуальной машиной - делайте это внутри тестов". При этом в начальном состоянии виртуальной машины даже не существует - весь процесс создания и разворачивания виртуального стенда делается автоматически. С одной стороны, это даёт большое преимущество: все, что нужно иметь на компьютере для прогона тестов с нуля - это набор установочных iso-образов и файл с тестовым скриптом. Как следствие - тесты можно легко перемещать между компьютерами. С другой стороны, очевидный минус такого подхода - необходимость реализовывать в тестовых сценариях все подготовительные действия, даже установку стандартизированных ОС.

Очевидно, что в таких условиях прогон тестов "с нуля" каждый раз становится крайне неэффективным - ведь подготовка виртуального стенда может занимать по времени гораздо больше времени, чем собственно проведение значимых тестов.

Поэтому в платформе Testo большое внимание уделяется вопросу экономии времени при прогоне тестов в тех случаях, когда это возможно.

Основная идея заключается в следующем: при первом запуске тестов (когда даже виртуальные машины еще не созданы) все эти тесты прогоняются с нуля, от начала до конца - в полном объёме. Каждый успешный тест при этом кешируется. Это означает следующее:

1. Для всех виртуальных машин и виртуальных флешек создаются физические снепшоты, которые позволяют "запомнить" состояние этих сущностей на момент окончания теста (только если у теста не указан атрибут `no_snapshots`, который мы рассмотрим позднее);
2. Для теста создаётся набор метаданных, которые хранятся на диске и позволяют зафиксировать множество различных факторов, которые затем будут помогать при оценке актуальности кеша.

Если тест по какой-то причине "свалился", то кеш для него не создаётся и все тесты, которые от него зависят (тесты-потомки), автоматически помечаются как проваленные.

Понятно, что первый прогон тестов может быть достаточно долгим. В качестве аналогии можно привести пример компиляции "с нуля" очень объёмного проекта. Но затем, конечно, благодаря инкрементальной компиляции процесс повторной сборки проекта занимает гораздо меньше времени, т.к. перекомпилируются только те объектные файлы, в исходниках которых произошли какие-то изменения. В платформе Testo используется похожий подход.

При повторном запуске тестов платформа Testo сначала оценивает актуальность кеша уже прошедших успешных тестов. В этом процессе задействовано много [факторов](/docs/lang/test#проверка-кеша), среди которых основные это:

1. Целостность самих тестовых сценариев (причем незначимые изменения не учитываются);
2. Целостность конфигураций виртуальных машин и флешек, участвующих в тесте;
3. Целостность файлов, которые попадают внутрь виртуальной машины с помощью действия `copyto`

Если кеш признаётся актуальным, то тест заново прогоняться не будет. Если вообще все тесты имеют актуальный кеш, то ни один тест не будет выполнен (по аналогии с инкрементальной компиляцией, когда исходники вообще не поменялись). Если же кеш признаётся недействительным, то сам тест **и все его потомки** становится в очередь на выполнение.

При этом помеченный на выполнение тест будет опираться на результат работы своего непосредственного родителя (или родителей). Т.к. у успешно пройденных родительских тестов создаются снепшоты всех виртуальных машин и флешек, то Testo получает возможность просто восстановить их состояние и продолжить с того же места, на котором закончился родительский тест.

Если просуммировать всё вышесказанное, то получается следующая картина:

1. Первый раз тесты прогоняются достаточно долго, т.к. выполняются все "подготовительные" тесты: установка ОС, настройка IP-адресов и прочее;
2. При повторных запусках будут прогоняться только тесты, в которых "что-то изменилось". Т.к. в подготовительных тестах очень редко когда что-то меняется, то они останутся закешированными и повторно прогоняться не будут.

## С чего начать?

В прошлой части мы столкнулись с ситуацией, когда все тесты оказались закешированы и не стали прогоняться даже после того, как мы заменили все вхождения строковых констант на обращение к параметрам. Чтобы понять, почему так произошло, давайте рассмотрим пример.

Для примера возьмем набор тестов, которые мы сделали в предыдущей части

![](/static/tutorials/5_caching/tests_tree.png)

Для начала необходимо прогнать все тесты и добиться того, чтобы они были закешированы (обратите внимание, запуск происходит без аргумента `invalidate`). Если какие-то тесты у вас были незакешированы, то пусть они выполнятся, а затем снова запустите testo с теми же аргументами.

В конечном счете вы должны увидеть такой вывод

<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
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

Теперь давайте попробуем поэкспериментировать с нашими тестовыми сценариями и посмотрим, что будет происходить с кешем.

Для начала давайте попробуем изменить тест `guest_additions_demo` и снова выполнить `testo`

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		#Измененный скрипт
		exec bash """
			echo Modified Hello world
			echo from bash
		"""
		#Двойные кавычки внутри скриптов необходимо экранировать
		exec python3 "print(\"Hello from python3!\")"
	}
}
```

Вывод будет следующим:

<Terminal height="650px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
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
	<span className=" ">+ echo Modified Hello world<br/></span>
	<span className=" ">Modified Hello world<br/></span>
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

Ожидаемо, изменение тестового сценария привело к потере кеша в этом тесте. При этом можно заметить, что запустился только дочерний тест `guest_additions_demo`, причем для его проведения состояние виртуальной машины `my_ubuntu` было восстановлено из снепшота `guest_additions_installation`, который создался ранее во время успешного прогона соответствуующего теста.

При этом, добавление пустых строк, отступов и комментариев не влияет на целостность кеша. Попробуйте добавить или убрать несколько комментариев или пустых строк/оступов и убедитесь в этом сами.

Давайте теперь рассмотрим тест `guest_additions_installation`. Давайте в этом тесте попробуем параметризировать строку с установкой deb-пакета

```testo
param guest_additions_pkg "*.deb"
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
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
}
```

и снова выполним тестовый сценарий. Мы увидим следующий вывод

<Terminal>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
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

То есть все тесты являются закешированными, несмотря на то, что мы только что изменили текст одного из тестов. Похожую ситуацию мы наблюдали в прошлой части обучения после введения параметров. Почему же так происходит?

Дело в том, что в платформе Testo при подсчете контрольных сумм комманд в тестах учитываются только **итоговые** значения всех строк уже после подстановки значений параметров в целевые строки. До применения параметра наше действие выглядело как `type "dpkg -i /media/*.deb"`, и после подстановки параметра действие будет выглядеть точно так же. Поэтому кеш считается целостным, несмотря на изменение текста самого сценария. Аналогичная ситуация была и в предыдущем уроке.

Однако, давайте попробуем поменять значение параметра

```testo
param guest_additions_pkg "testo-guest-additions*"
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		...
	}
}
```

И попробуем запустить сценарий. Если вы не хотите каждый раз подтверждать своё согласие с тем, что кеш теста был потерян, вы можете использовать аргумент `--assume_yes`

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
	<span className="blue ">[ 33%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 33%] Restoring snapshot </span>
	<span className="yellow ">ubuntu_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
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
	<span className="green bold"> PASSED in 0h:0m:17s<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo Modified Hello world<br/></span>
	<span className=" ">Modified Hello world<br/></span>
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
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:21s<br/></span>
	<span className="blue bold">UP-TO-DATE: 1<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Очевидно, что наш тест `guest_additions_installation` потерял свой кеш, т.к. значение строки с параметром `guest_additions_pkg` изменилось. Здесь примечательно то, что вместе с потерей кеша в тесте `guest_additions_installation` автоматически был сброшен кеш производного теста `guest_additions_demo`, хотя никаких изменений в нем мы не проводили.


## Целостность файлов в `copyto` и `plug dvd`

В третьей части мы упоминали о том, что после установки гостевых дополнений пользователю становится доступно несколько новых действий, в том числе действие `copyto`, позволяющее копировать файлы из хостовой машины в виртуальную в ходе самих тестов. Давайте познакомимся с этим действием поближе.

Допустим, в ходе тестового сценария нам необходимо передать небольшой текстовый файл внутрь виртуальной машины. Давайте создадим такой файл в той же папке, что и наш тестовый сценарий `hello_world.testo`

Сам тестовый сценарий необходимо подкорректировать

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		#Измененный скрипт
		exec bash """
			echo Modified Hello world
			echo from bash
		"""
		#Двойные кавычки внутри скриптов необходимо экранировать
		exec python3 "print(\"Hello from python3!\")"

		copyto "./testing_copyto.txt" "/tmp/testing_copyto.txt"
		exec bash "cat /tmp/testing_copyto.txt"
	}
}
```

<Terminal height="700px">
	<span className="">user$ echo "This should be copied inside my_ubuntu" &gt; ~/testo/testing_copyto.txt<br/></span>
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --assume_yes<br/></span>
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
	<span className=" ">+ echo Modified Hello world<br/></span>
	<span className=" ">Modified Hello world<br/></span>
	<span className=" ">+ echo from bash<br/></span>
	<span className=" ">from bash<br/></span>
	<span className="blue ">[ 67%] Executing python3 command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">Hello from python3!<br/></span>
	<span className="blue ">[ 67%] Copying </span>
	<span className="yellow ">./testing_copyto.txt </span>
	<span className="blue ">to virtual machine </span>
	<span className="yellow ">my_ubuntu </span>
	<span className="blue ">to destination </span>
	<span className="yellow ">/tmp/testing_copyto.txt </span>
	<span className="blue ">with timeout 10m<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ cat /tmp/testing_copyto.txt<br/></span>
	<span className=" ">This should be copied inside my_ubuntu<br/></span>
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

В действии `copyto` необходимо указать копируемый файл (т.к. он лежит в одной папке с тестовым сценарием, то достаточно указать относительный путь `./`), а также **полный путь, включая имя конечного файла** куда необходимо скопировать этот файл.

Попробуйте выполнить этот сценарий и убедитесь, что он закешировался.

Далее идёт важный момент. Как уже упоминалось, при рассчитывании кеша тестов во внимание принимается также целостность файлов, которые участвуют в тесте (в том числе в действии `copyto`). Причём целостность высчитывается по специальному алгоритму:

1. Если копируемый файл размером меньше одного мегабайта, то высчитывается целостность **содержимого** этого файла;
2. Если копируемый файл размером больше одного мегабайта, то высмчитывается целостность **даты последнего изменения** этого файла;
3. Если копируется папка, то при первые два правила применяются к **каждому файлу по отдельности**, размер файлов внутри папки не суммируется.

Попробуйте в этом убедиться. Т.к. наш файл `testing_copyto.txt` меньше одного мегабайта, то к нему применяется первый шаг алгоритма. Попробуйте изменить дату последнего изменения файла `testing_copyto.txt` и убедитесь, что тест остался закешированным. И при этом изменение содержимого файла тут же приведет к сбросу кеша.

Можете также создать большой файл (больше одного мегабайта) и убедиться, что его целостность высчитывается только на основе даты последнего изменения.

> Такое же поведение при подсчете целостности относится к проверке iso-образов в действии `plug dvd`, а также при загрузке папки на виртуальную флешку. Виртуальные флешки мы рассмотрим позднее

> Есть и другие факторы, влияющие на кеш тестов. Например, конфигурация виртуальных сущностей: сетей, флешек и машин. Всю эту информацию можно найти в [документации](/docs/lang/test#проверка-кеша). В частности, целостность iso-браза в атрибуте `iso` виртуальной машины тоже влияет на закешированность теста


## Ручной сброс кеша

Конечно, в платформе Testo предусмотрена возможность вручную сбросить кеш у теста (или даже группы тестов). Для этого существует аргумент командной строки `invalidate`, который по своему формату совпадает с `--test_spec`, и позволяет указывать шаблон имён тех тестов, кеш которых надо сбросить.

Например, если вы хотите сбросить кеш всех тестов, связанных с гостевыми дополнениями, выполните следующую команду


<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --invalidate guest_additions*<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">ubuntu_installation<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">guest_additions_installation<br/></span>
	<span className="magenta ">guest_additions_demo<br/></span>
	<span className="blue ">[ 33%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_installation<br/></span>
	<span className="blue ">[ 33%] Restoring snapshot </span>
	<span className="yellow ">ubuntu_installation</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
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
	<span className="green bold"> PASSED in 0h:0m:17s<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">guest_additions_demo<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo Modified Hello world<br/></span>
	<span className=" ">Modified Hello world<br/></span>
	<span className=" ">+ echo from bash<br/></span>
	<span className=" ">from bash<br/></span>
	<span className="blue ">[ 67%] Executing python3 command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">Hello from python3!<br/></span>
	<span className="blue ">[ 67%] Copying </span>
	<span className="yellow ">./testing_copyto.txt </span>
	<span className="blue ">to virtual machine </span>
	<span className="yellow ">my_ubuntu </span>
	<span className="blue ">to destination </span>
	<span className="yellow ">/tmp/testing_copyto.txt </span>
	<span className="blue ">with timeout 10m<br/></span>
	<span className="blue ">[ 67%] Executing bash command in virtual machine </span>
	<span className="yellow ">my_ubuntu</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ cat /tmp/testing_copyto.txt<br/></span>
	<span className=" ">This should be copied inside my_ubuntu<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">guest_additions_demo</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">my_ubuntu<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">guest_additions_demo</span>
	<span className="green bold"> PASSED in 0h:0m:4s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:21s<br/></span>
	<span className="blue bold">UP-TO-DATE: 1<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Итоги

Кеширование является важным механизмом платформы Testo и позволяет существенно экономить время на повторных прогонах тестов. Кеширование позволяет вам запускать только те тесты, которые необходимо запустить в связи с какими-то значимыми изменениями.

Итоговый скрипт можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/5)
