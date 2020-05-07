# Часть 14. Циклы

## С чем Вы познакомитесь

В этом уроке вы познакомитесь с циклами в языке `testo-lang`.

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
5. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [тринадцатой части](13_js)

## Вступление

Наш цикл ознакомительных уроков с платформой Тесто плавно подходит к своему завершению, и в языке `testo-lang` осталась лишь одна возможность, с которой мы ещё не познакомились - это циклы.

До сих пор мы создавали тестовые сценарии без использования циклов - они нам были ни к чему и скорее бы только мешали. Однако, конечно, рано или поздно вы можете столкнуться с задачей, когда вам потребуется сделать несколько однотипных действий - и в этом случае вам очень пригодятся циклы.

Циклы могут также использоваться и в другом аспекте - при создании вспомогательных макросов по выбору чего-либо на экране когда вы заранее не знаете (или не хотите тратить время чтобы узнать) сколько именно действий вам надо совершить, чтобы дождаться нужного вам события.

Например, давайте посмотрим на уже знакомый нам экран выбора языка при установке Ubuntu Server

![Language](/static/tutorials/13_js/language.png)

Конечно, мы исходили из того, что нам требуется английский язык, и поэтому просто нажимали клавишу Enter, т.к. этот пункт уже выделен по-умолчанию. Но что, если нам требуется другой пункт, например "Русский"? Для этого нам надо 29 раз нажать на клавишу Down, чтобы для начала выделить эту надпись. А что, если мы не хотим считать нужное количество нажатий на клавишу Down? Если мы хотим просто вызвать макрос `select_menu("Русский")` и не беспокоиться о том, сколько раз для этого надо нажимать клавишу вниз? В языке `testo-lang` вы можете создавать такие макросы, и в этом вам помогут циклы, о которых мы сегодня и поговорим.

## С чего начать?

В этом уроке мы попробуем упростить работу в консольном файловом менеджере [Vifm](https://www.tecmint.com/vifm-commandline-based-file-manager-for-linux/) с помощью циклов. Мы выбрали именно этот файловый менеджер потому что его интерфейс больше всего подходит для простого примера применения циклов. При определенном желании вы можете проделать такую же работу и с другими файловыми менеджерами по своему усмотрению.

Для начала давайте почитсим наше дерево тестов от всего ненужного, оставим только одну виртуальную машину, установку гостевых дополнений для неё и установку  менеджера Vifm из репозиртория. В последнем тесте мы будем автоматизировать работу с менеджером Vifm, а пока ограничимся только его запуском.

```testo
test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}

test server_prepare: server_install_ubuntu {
	server {
		install_guest_additions("${server_hostname}", "${server_login}")
		exec bash "apt install -y vifm"
	}
}

test cycles_demo: server_prepare {
	server {
		type "vifm /"; press Enter
		abort "stop here"
	}
}
```

После выполнения этого теста мы увидим интерфейс менеджера Vifm

![Vifm](/static/tutorials/14_for/Vifm.png)

Давайте представим, что нам необходимо зайти в директорию `/usr/sbin`.  Для этого нам нужно сначала 19 раз нажать "Down", затем Enter, затем снова 6 раз Enter, и наконец снова Enter. Напишем это в сценарии

```testo
test cycles_demo: server_prepare {
	server {
		type "vifm /"; press Enter
		sleep 2s

		press Down*19, Enter
		press Down*6, Enter
		abort "stop here"
	}
}
```

Такой скрипт действительно приводит нас к нужному результату 

![usr_sbin](/static/tutorials/14_for/usr_sbin.png)

Но выглядит это всё не очень здорово, не так ли? Здесь есть две глобальные проблемы:

1. Скрипт абсолютно плохо читается - тяжело понять, куда мы хотели бы попасть после того, как спустились вниз на 19 пунктов, а затем снова на 6;
2. Каждый раз высчитывать точное количество нажатий на клавиши вниз-вверх - задача практически невыполнимая, да и бесмыссленная.

На самом деле, решить проблему №1 можно, вспомнив [предыдущий урок](13_js), посвященный JS-селекторам (в частности, определению цвета фона у надписей). В самом деле, мы можем каждый раз осуществлять проверку, что мы выбираем именно тот пункт меню, который нам действительно нужен. Для этого лишь надо формализовать понятие "Выделенный пункт меню". В случае с Vifm выделенная строка отличается от любой другой надписи синим фоном, поэтому давайте вспомним предыдущий урок и модифицируем наш простой тест

```testo
macro vifm_select(menu_entry) {
	if (check js "find_text().match('${menu_entry}').match_background('blue').size() == 1") {
		press Enter
	} else {
		abort "Menu entry ${menu_entry} is not selected"
	}
}

test cycles_demo: server_prepare {
	server {
		type "vifm /"; press Enter
		sleep 2s

		press Down*19; vifm_select("usr")
		press Down*6;  vifm_select("sbin")
		abort "stop here"
	}
}
```

Теперь наш тест выглядит уже немного более читаемым (теперь мы хотя бы можем понять, какой пункт меню мы выбираем), но у нас всё еще остается проблема с тем, что нам придется каждый раз высчитывать точное количество перещёлкиваний Вниз-Вверх чтобы выделить нужный пункт меню. Эту проблему можно исправаить с помощью цикла.

## Цикл для выбора пункта меню

Давайте модифицируем наш макрос `vifm_select` так, чтобы он выбирал нужный пункт меню "за нас". То есть автоматически определял количество нажатий на клавишу "Вниз", необходимое для перемещения на нужную строчку. Мы построим алгоритм следующим образом:

1. Сначала надо переместить "указатель" на самый верхний пункт. Для этого в Vifm существует комбинация клавиш `gg`;
2. Теперь мы будем в цикле последовательно проверять, выделен ли нужный пункт меню;
3. Если нужный пункт не выделен, необходимо переместиться на одну позицию вниз и снова произвести проверку;
4. Если нужный пункт выделен, мы нажимаем на клавишу Enter и выходим из цикла;
5. Если мы израсходовали количество попыток на поиск нужного пункта, мы высвечиваем сообщение об ошибке, что нужного пункта меню не существует.

В коде это будет выглядеть следующим образом

```testo
macro vifm_select(menu_entry) {
	press g*2

	for (i IN RANGE "0" "50") {
		if (check js "find_text().match('${menu_entry}').match_background('blue').size() == 1") {
			print "Got to entry ${menu_entry} in ${i} steps"
			press Enter
			break
		}
		press Down
		sleep 50ms
	} else {
		abort "Menu entry ${menu_entry} doesn't exist!"
	}
}
```

Давайте разберем синтаксический формат цикла `for`. Мы видим, что в заголовке используется выражение `i IN RANGE "0" "50"`. Это выражение означает, что тело цикла необходимо выполнить 50 раз (левая граница RANGE включается в отрезок, а правая - нет). `i` выступает счетчиком, к которому можно обращаться внутри цикла. Мы воспользовались этой возможностью чтобы вывести вспомогательную информацию на консоль (с помощью действия [`print`](/docs/lang/actions#print)), за сколько шагов мы смогли добраться до нужного пункта меню.

Мы выбрали число 50 как некий теоритический предел - количество шагов, за которое **точно** можно добраться до нужного пункта меню. Однако если количество файлов внутри директории превышает 50, то этот макрос может не сработать. Выбирайте верхний предел для циклов разумно - слишком большая верхняя граница приведёт к тому, что в случае ошибки вы узнаете об этой ошибке очень нескоро.

Мы также вставили небольшой `sleep` после нажатия на `Down` чтобы дать немного времени графическому интерфейсу отреагировать на наше воздействие. 

Обратите внимание на секцию `else` для цикла. Подобно некоторым языкам, в `testo-lang` есть возможность определять секцию `else` на тот случае, если прошли **все запланированные** итерации цикла. В нашем случае все пройденные операции цикла означают, что нужный пункт меню так и не был найден.

>В том случае, когда нижняя граница для счетчика равна нулю, можно сделать запись немного более компактной: `i IN RANGE 50"`

>В платформе Testo отсутствует возможность создавать бесконечные циклы, т.к. это противоречит самой природе тестирования: тесты обязательно должны рано или поздно закончиться

Давайте посмотрим на сам тест

```testo
test cycles_demo: server_prepare {
	server {
		type "vifm /"; press Enter
		sleep 2s
		vifm_select("usr")
		vifm_select("sbin")
		abort "stop here"
	}
}
```

Мы видим, что он стал намного компактнее и выразительнее. Теперь для навигации внутри Vifm достаточно просто писать вызовы макроса `vifm_select`, не заботясь о том, какое количество раз нужно нажимать клавиши "Вверх" - "Вниз" для того, чтобы добраться до необходимого пункта.

Если запустить наш тест, то вот что мы увидим:

<Terminal height="600px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec cycles_demo<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"vifm /" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 2s<br/></span>
	<span className="blue ">[ 67%] Calling macro </span>
	<span className="yellow ">vifm_select(</span>
	<span className="yellow ">menu_entry="usr"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">G </span>
	<span className="blue ">2 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] </span>
	<span className="yellow ">server</span>
	<span className="blue ">: Got to entry usr in 19 steps<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Calling macro </span>
	<span className="yellow ">vifm_select(</span>
	<span className="yellow ">menu_entry="sbin"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">G </span>
	<span className="blue ">2 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] </span>
	<span className="yellow ">server</span>
	<span className="blue ">: Got to entry sbin in 6 steps<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="red bold">/home/alex/testo/tests.testo:36:3: Caught abort action on virtual machine server with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">cycles_demo</span>
	<span className="red bold"> FAILED in 0h:0m:32s<br/></span>
	<span className="">user$ </span>
</Terminal>

Вы можете заметить, что тест стал прогоняться гораздо дольше - это связано с резко возросшим количеством проверок состояния экрана. Если вам захочется снова его немного ускорить.
После первого долгого прогона мы знаем (благодаря логированию с помощью `print`), что, например, для достижения пункта `usr` требуется 19 нажатий клавиши вниз. Мы могли бы "подсказать" макросу, вручную передвинув "курсор" туда, куда нужно.

```testo
macro vifm_select(menu_entry) {
	if (check js "find_text().match('${menu_entry}').match_background('blue').size() == 1") {
		print "Entry ${menu_entry} is already selected"
		press Enter
	} else {
		press g*2

		for (i IN RANGE "0" "50") {
			if (check js "find_text().match('${menu_entry}').match_background('blue').size() == 1") {
				print "Got to entry ${menu_entry} in ${i} steps"
				press Enter
				break
			}
			press Down
			sleep 50ms
		} else {
			abort "Menu entry ${menu_entry} doesn't exist!"
		}
	}
}

test cycles_demo: server_prepare {
	server {
		type "vifm /"; press Enter
		sleep 2s
		press Down*19 #Необязательная подсказка
		vifm_select("usr")
		vifm_select("sbin")
		abort "stop here"
	}
}
```

Наш макрос теперь перед какими-либо действиями сначала проверяет, что нужный пункт ещё не выделен. Если пункт сразу выделен, то нажимается Enter и ничего больше не происходит. Если же пункт не выделен, то начинается стандартная процедура.

<Terminal height="600px">
	<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec cycles_demo<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">cycles_demo<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"vifm /" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 2s<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">19 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Calling macro </span>
	<span className="yellow ">vifm_select(</span>
	<span className="yellow ">menu_entry="usr"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('usr').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] </span>
	<span className="yellow ">server</span>
	<span className="blue ">: Entry usr is already selected<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Calling macro </span>
	<span className="yellow ">vifm_select(</span>
	<span className="yellow ">menu_entry="sbin"</span>
	<span className="yellow ">)</span>
	<span className="blue "> in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">G </span>
	<span className="blue ">2 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 50ms<br/></span>
	<span className="blue ">[ 67%] Checking </span>
	<span className="yellow ">find_text().match('sbin').match_background('blue').size() == 1 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] </span>
	<span className="yellow ">server</span>
	<span className="blue ">: Got to entry sbin in 6 steps<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="red bold">/home/alex/testo/tests.testo:42:3: Caught abort action on virtual machine server with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">cycles_demo</span>
	<span className="red bold"> FAILED in 0h:0m:13s<br/></span>
	<span className="">user$ </span>
</Terminal>

Заметим, что время прогона теста существенно уменьшилось - с 32 секунд до 13 (в 2,5 раза).

## Итоги

Циклы в языке `testo-lang` могут существенно упростить написание тестов, связанных с неопределенным заранее количеством действий. Самое главное - грамотно определить условие того, что нужное событие наступило, а также правильно выбрать инкрементальные шаги, которые помогут вам достичь этого события.

Несмотря на то, что циклы чаще всего работают не слишком быстро, вы чаще всего можете "ускорить" такие циклы, подобрав правильные "подсказки". Подсказки можно получить, залогировав один раз "долгий" прогон.

Готовые скрипты можно найти [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/14)