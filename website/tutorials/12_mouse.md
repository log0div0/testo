# Часть 12. Управление мышкой

## С чем Вы познакомитесь

В этом уроке вы познакомитесь с основами управления мышью в языке `testo-lang`

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. Имеется установочный образ [Ubuntu Desktop 18.04](https://releases.ubuntu.com/18.04.4/ubuntu-18.04.4-desktop-amd64.iso) с расположением `/opt/iso/ubuntu_desktop.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
4. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.

## Вступление

В платоформе Testo реализована возможность управлять мышкой в тех случаях, когда она доступна. Сначала этот процесс может показаться несколько запутанным, но как только вы поймёте основные [принципы](/docs/lang/mouse#надпись-на-экране-с-дополнительными-уточнениями) позиционирования курсора на экране, вы уже не сможете разучиться это делать. Это как езда на велосипеде - достаточно один раз прочуствовать.

В качестве тренировки мы будем использовать Ubuntu Desktop 18.04. Если точнее - мы попытаемся автоматизировать установку этой ОС, используя мышку везде, где это возможно.

## С чего начать?

Давайте оставим пока наш стенд, который мы постепенно совершенствовали в предыдущих уроках, и создадим новый тестовый проект. Назовём его `mouse.testo`, и в нём будет располагаться весь необходимый нам код (этого кода будет не так много, поэтому разносить проект по разным файлам особого смысла нет).

Начнём с уже знакомой заготовки

```testo
network internet {
	mode: "nat"
}

machine ubuntu_desktop {
	cpus: 1
	ram: 2Gb
	iso: "${ISO_DIR}/ubuntu_desktop.iso"
	
	disk main: {
		size: 20Gb
	}

	nic internet: {
		attached_to: "internet"
		adapter_type: "e1000"
	}
}

param login "desktop"
param hostname "desktop-PC"
param password "1111"

test install_ubuntu {
	ubuntu_desktop {
		start
		abort "stop here"
	}
}
```

Создаём новую виртуальную машину `ubuntu_desktop`. Из-за наличия графического интерфейса ей потребуется чуть больше оперативной памяти, чем `ubuntu_server` из предыдущих уроков. Также можно выделить ей побольше места на диске по тем же причинам (10 Гб хватит). Заранее объявляем знакомые параметры и переходим к написанию самого теста по установке ОС.

Первые два экрана ещё не содержат никакой графической составляющей, поэтому мы не будем на них долго останавливаться. Давайте сразу допишем тест так, чтобы попасть на первый экран с графическим интерфейсом:

```testo
test install_ubuntu {
	ubuntu_desktop {
		start
		wait "English"
		press Enter
		wait "Try Ubuntu without installing"; press Down, Enter
		wait "Welcome" timeout 5m
		abort "stop here"
	}
}
```

<Terminal height="400px">
	<span className="">user$ sudo testo run mouse.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Try Ubuntu without installing </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Welcome </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="red bold">/home/alex/testo/mouse.testo:28:3: Caught abort action on virtual machine ubuntu_desktop with message: stop here<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">install_ubuntu</span>
	<span className="red bold"> FAILED in 0h:0m:46s<br/></span>
	<span className="">user$ </span>
</Terminal>

Мы увидим такой экран

![Welcome](/static/tutorials/12_mouse/welcome.png)

> Обратите внимание, что иногда установка `Ubuntu` начинается по другому сценарию - первым появляется графический экран с предложением выбрать `Try Ubuntu` или `Install Ubuntu`. Попробуйте использовать свои знания из предыдущих уроков и модифицировать скрипт таким образом, чтобы он учитывал оба сценария начала установки.

Очевидно, что для продолжения установки нам необходимо нажать на кнопку `Continue`. Для этого мы воспользуемся действием [`mouse click`](/docs/lang/mouse#mouse-click(lckick,-rclick,-dclick))

```testo
test install_ubuntu {
	ubuntu_desktop {
		start
		wait "English"
		press Enter
		wait "Try Ubuntu without installing"; press Down, Enter
		wait "Welcome" timeout 5m
		mouse click "Continue"
		abort "stop here"
	}
}
```

Действие `mouse click "Continue"` означает примерно следующее: "Дождись появления на экране надписи `Continue`" (По умолчанию ожидание составляет одну минуту), перемести указатель мыши на центр этой надписи и выполни клик по левой кнопке мышки". Т.к. надпись "Continue" уже и так будет находиться на экране на момент вызова действия `mouse click`, то действие отработает сразу же.

Выглядит несложно, правда? Чаще всего действия с мышкой будут выглядеть именно так - достаточно просто и прямолинейно. Однако механизм управления мышкой в `testo-lang` гораздо мощнее, чем может показаться на первый взгляд. В этом мы сейчас и убедимся.

После наших изменений мы увидим следующий по счёту экран:

![Keyboard Layout](/static/tutorials/12_mouse/keyboard_layout.png)

Давайте заметим следующий очень важный момент. Как мы уже говорили. действие `mouse click "Continue"` предписывает переместить курсор мыши на центр надписи "Continue". Действие успешно отработало, и теперь нам надо ещё раз продублировать действие `mouse click "Continue"`

```testo
test install_ubuntu {
	ubuntu_desktop {
		start
		wait "English"
		press Enter
		wait "Try Ubuntu without installing"; press Down, Enter
		wait "Welcome" timeout 5m
		mouse click "Continue"
		wait "Keyboard layout"
		mouse click "Continue"
		abort "stop here"
	}
}
```

Но реальность такова, что повторное действие `mouse click "Continue"` может и не сработать! Дело в том, что теперь сам курсор мыши частично загораживает надпись `Continue`, и теперь платформа Testo может просто не найти на экране нужную надпись `Continue`! Не найдя нужную надпись за одну минуту (таймаут по умолчанию), действие `mouse click` приведёт к ошибке.

Для того, чтобы избежать этой ситуации, необходимо сделать так, чтобы курсор мыши не загораживал надпись `Continue` и позволил успешно отработать механизму распознавания текста на экране в платформе Testo. Существует несколько способов это сделать, и в этом уроке мы рассмотрим два из них.

## Позиционирование курсора внутри надписи

Итак, наша задача звучит так: на экране `Welcome` необходимо кликнуть на кнопку `Continue` так, чтобы курсор не загораживал надпись `Continue` на следующем экране `Keyboard layout`. Для этого мы прибегнем к помощи [уточнения](/docs/lang/mouse#надпись-на-экране-с-дополнительными-уточнениями) позиционирования курсора внутри найденного объекта.

По умолчанию, действие `mouse click "Continue"` перемещает курсор в центр надписи `Continue`. Для того, чтобы курсор не загораживал надпись после клика, нам необходимо пересетить его не в центр надписи, а так, чтобы он не мешался. В языке testo-lang существует возможность изменить позиционирование курсора внутри найденного объекта. Например, мы можем переместить курсор в нижний край надписи `Continue`. Эту работу выполняет спецификатор `center_bottom` (что означает - центр по оси Х и нижняя граница по оси Y). Его использование выглядит так:

```testo
test install_ubuntu {
	ubuntu_desktop {
		start
		wait "English"
		press Enter
		wait "Try Ubuntu without installing"; press Down, Enter
		wait "Welcome" timeout 5m
		mouse click "Continue".center_bottom()
		wait "Keyboard layout"
		abort "stop here"
	}
}
```

Теперь экран `Keyboard layout` будет выглядеть уже так

![Keyboard Layout 2](/static/tutorials/12_mouse/keyboard_layout_2.png)

Как мы видим, теперь курсор мыши не загораживает надпись `Continue` и мы можем легко повторно нажать на эту надпись

```testo
test install_ubuntu {
	ubuntu_desktop {
		start
		wait "English"
		press Enter
		wait "Try Ubuntu without installing"; press Down, Enter
		wait "Welcome" timeout 5m
		mouse click "Continue".center_bottom()
		wait "Keyboard layout"
		mouse click "Continue".center_bottom()
		abort "stop here"
	}
}
```

На следующем экране

![Updates](/static/tutorials/12_mouse/updates.png)

Мы выберем минимальную инсталяцию и отключим загрузку обновлений во время установки.

```testo
wait "Updates and other software"
mouse click "Minimal installation"
mouse click "Download updates while installing"; mouse click "Continue"
```

Заметьте, мы намеренно сделали так, чтобы после нажатия на `Continue` курсор мыши снова загораживал надпись на следующем экране. Это мы сделали для демонстрации ещё одного способа "убрать" курсор мыши с картинки так, чтобы он не мешался.

## Перемещение курсора по координатам

Дойдя до экрана с выбором типа установки мы снова видим, что курсор перегородил надпись, которую мы хотим кликнуть. 

![Installation type](/static/tutorials/12_mouse/installation_type.png)

Для того, чтобы убрать курсор мыши, не обязательно "подгадывать" с помощью спецификаторов, где он может или должен оказаться. Можно просто подвинуть курсор в сторону, используя абсолютные координаты.

```testo
mouse click "Download updates while installing";
mouse click "Continue"
wait "Installation type";
mouse move 0 0; 
mouse click "Install Now".center_bottom()
```
Здесь мы использовали действие `mouse move`, которое работает точно так же, как и `mouse click`, но вместо выполнения кликов просто перемещает курсор мыши в заданное место. Перемещать курсор можно как с помощью указания надписи (`mouse move "Continue"`, например), так и с помощью передачи абсолютных [координат](/docs/lang/mouse#координаты).

Абсолютные координаты начинают свой отсчёт с вернего левого угла экрана. В этой точке координаты по обоим осям равны 0. Чтобы переместить указатель в правый или нижний угод необходимо знать, какое такущее разрешение экрана. Например, при разрешении 800х600 правый нижний угол экрана будет иметь координаты 799х599. Поведение не определено в случае попыток передвинуть указатель "за пределы" экрана.

`mouse move 0 0` по большей части означает "Передвинуть указатель в левый вернхий угол экрана, чтобы он не мешался".

## Выбор нужного экземпляра надписи

Теперь перед нами появляется такой экран.

![Write changes](/static/tutorials/12_mouse/write_changes_to_disk.png)

Конечно, нам необходимо нажать на `Continue`. Однако, если мы попробуем это сделать привычным нам способом:

```testo
mouse click "Download updates while installing"; mouse click "Continue"
wait "Installation type"; mouse move 0 0; mouse click "Install Now".center_bottom()
wait "Write the changes to disks?";
mouse click "Continue".center_bottom()
```

То мы увидим, что это приводит к ошибке

<Terminal height="620px">
	<span className="">user$ sudo testo run mouse.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">install_ubuntu<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Try Ubuntu without installing </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">DOWN </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Welcome </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Continue.center_bottom() </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Continue.center_bottom() </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Updates and other software </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Minimal installation </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Download updates while installing </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Continue </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Installation type </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse moving </span>
	<span className="blue ">on coordinates </span>
	<span className="yellow ">0 0 </span>
	<span className="blue ">in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Install Now.center_bottom() </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Write the changes to disks? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="blue ">[  0%] Mouse clicking </span>
	<span className="blue ">on </span>
	<span className="yellow ">Continue.center_bottom() </span>
	<span className="blue ">with timeout 1m in virtual machine </span>
	<span className="yellow ">ubuntu_desktop<br/></span>
	<span className="red bold">/home/alex/testo/mouse.testo:42:45: Error while performing action click Continue.center_bottom() on virtual machine ubuntu_desktop:<br/>	-Can't apply specifier "center_bottom": there's more than one object<br/></span>
	<span className="red bold">[100%] Test </span>
	<span className="yellow bold">install_ubuntu</span>
	<span className="red bold"> FAILED in 0h:1m:11s<br/></span>
	<span className="">user$ </span>
</Terminal>

С чем же это связано? Дело в том, что на этом экране две надписи `Continue`:

![Write changes 2](/static/tutorials/12_mouse/write_changes_to_disk_2.png)

Несмотря на то, что одна надпись начинается с заглавной буквы "С", а другая - с прописной, механизм распознавания изображений иногда может расценивать такие надписи как одинаковые (хотя механизм распознавания строк изначально чувствителен к регистру). Такое иногда случается, и ничего страшного в этом нет.

В чем же проблема, почему наша попытка применить спецификатор `center_bottom` привела к ошибке? Дело в том, что когда на экране несколько экземпляром надписи, на которую мы хотим кликнуть (или просто передвинуть курсор), платформа Testo не может быть уверена, какую именно надпись мы имеем в виду. Соответственно, т.к. платформа не может быть уверена, какую именно надпись мы имеем в виду, она не может применить к ней спецификатор `center_bottom` (о чем и свидетельствует текст ошибки).

Для того, чтобы исправить эту ошибку, достаточно воспользоваться спицификатором выбора конкретной надписи. Таких спецификаторов всего 4: `from_left()`, `from_right()`, `from_top()` и `from_bottom()`. Здесь мы воспользуемся спецификатором `from_bottom()`:

```testo
mouse click "Download updates while installing"; mouse click "Continue"
wait "Installation type"; mouse move 0 0; mouse click "Install Now".center_bottom()
wait "Write the changes to disks?";
mouse click "Continue".from_bottom(0).center_bottom()
```
Что же мы имеем в виду таким действием? Такое действие необходимо читать так:

1. Найди на экране все надписи `Continue` (если надписей вообще нет - дождись появления хотя бы одной надписи в течение 1 минуты);
2. Из всех найденных надписей выбери ту, которая ближе всего к нижней границе экрана;
3. К выбранной надписи примени спецификатор `center_bottom` - то есть передвинь курсор к нижней границе надписи по центру относительно оси Х;
4. После передвижения курсора выполни клик левой кнопкой мыши.

Раньше, когда необходимая надпись на экране у нас присутствовала в единственном экзкмпляре, этот уточняющий шаг можно было пропустить, но теперь он нам необходим. Если бы мы хотели кликнуть по верхней надписи `Continue`, мы бы использовали уточнение `from_bottom(1)` или `from_top(0)`. Конечно, если мы попытаемся обратиться к элементу `from_top(2)`, то получим ошибку, т.к. фактически "выйдем за пределы массива" доступных объектов.

## Финальное уточняющее позиционирование курсора

Очень скоро мы экран с предложением ввести логин и пароль.

![Who are you](/static/tutorials/12_mouse/who_are_you.png)

Конечно, мы можем ввести все данные, исопльзуя исключительно клавиатуру (для переключения между полями ввода достаточно нажимать клавишу Tab), но в показательных целях мы попробуем сделать это с помощью мышки.

Для начала наша цель будет заключаться в том, что нам нужно будет кликнуть на поле ввода нашего имени (Your name). Это поле ввода уже выделено по-умолчанию, но мы можем притвориться, что мы этого не знаем.

Для этого мы воспользуемся ещё одной возможностью уточнения позиционирования курсора в языке `testo-lang` - спецификатором `move`

```testo
wait "Where are you?"; mouse click "Continue".center_bottom()
wait "Who are you?";
mouse click "Your name:".right_center().move_right(20);
type "${login}"
```

В этом действии мы придержимаемся следующей логики:

1. Необходимо найти "точку отсчета" - некоторый объект на экране, от которого мы могли бы оттолкнуться при поиске необходимо поля ввода. В данном случае на эту роль отлично подойдёт надпись "Your name:". Т.к. это единственная надпись "Your name" на экране, мы можем пропустить уточнение `.from`;
2. Теперь нам необходимо спозиционировать курсор внутри найденной надписи. Т.к. необходимое поле ввода находится строго справа от надписи "Your name", то выглядит логичным пододвинуть курсор к правому краю найденной надписи. Для этого мы дописываем спецификатор `.right_center()`;
3. Но этого всё еще недостаточно, ведь курсор всё еще надохится не там, где нужно. Для того, чтобы наконец попасть курсором на нужное текстовое поле, нам необходимо сдвинуть мышку вправо на какое-то количество пикселей. Мы прикидываем, что 20 пикселей должно быть достаточно, и дописываем спецификатор `move_right(20)`. Этого должно быть полностью достаточно для того, чтобы курсор попал в необходимую область и мы смогли бы, наконец, выполнить клик;
4. Нужное поле ввода выделено, можно писать свой логин.

> Можно было бы пропустить уточнение `.right_center()` и написать сразу `mouse click "Your name:".move_right(20)`, но тогда отсчёт 20 пикселей начинался бы с центра надписи, а не с правого её края, что не очень удобно

> После применения спецификатора `.move_right(20)` не обязательно останавливаться - можно двигать курсор в любом направлении и сколько угодно. Например, `mouse click "Your name:".move_right(20).move_down(50).move_left(10).move_left(30)` и так далее

После того, как мы ввёдем свое имя (оно же логин), мы увидим ужасающее сгенерированное имя компьютера, которое нас совершенно не устраивает:

![Hostname](/static/tutorials/12_mouse/hostname.png)

Для того, чтобы его исправить, воспользуемся таким же приёмом:

```testo
wait "Where are you?"; mouse click "Continue".center_bottom()
wait "Who are you?";
mouse click "Your name:".right_center().move_right(20); type "${login}"
mouse click "Your computer's name".right_center().move_right(20); 
press LeftCtrl + A, Delete; type "${hostname}"
```
Отметим, что для удаления мы для начала должны выделить весь текст, используя сочетание CTRL+A.

А для записи пароля мы попробуем ориентироваться только на слово `Password` на экране и попробуем преминить все три спецификатора одновременно.

```testo
mouse click "Your computer's name".right_center().move_right(20); press LeftCtrl + A, Delete;  type "${hostname}"
mouse click "password:".from_top(0).right_center().move_right(20); type "${password}"
mouse click "password:".from_top(1).right_center().move_right(20); type "${password}"
```
Поздравляем, теперь вы познакомились с концепцией уточняющих спецификаторов при работе с мышкой в языке `testo-lang`! Давайте вкратце просуммируем основную логику применение уточнений:

1. Необходимо найти нужный экземпляр искомой надписи. Для этого используйте отсчеты от краёв экрана, а поможет вам в этом один из спецификаторов `from`. Если нужна вам надписаь находится на экране в единственном экземпляре, этот шаг можно пропустить.
2. Спозиционировать курсор внутри искомой надписи. Если вас устраивает позиционирование курсора по центру надписи, этот шаг можно пропустить.
3. Передвинуть курсор на некоторое количество пикселей относительно найденной точки. Если дополнительно двигаться курсор не нужно, этот шаг можно пропустить. Двигать можно влево-вправо-вверх-вниз неограниченное количество раз.

## Заканчиваем установку

С помощью действий `mouse` иногда можно сэкономить немного места в тестовом сценарии: в случае, когда вам нужно кликнуть на надпись, которая сама по себе является индикатором нужного вам события (то есть то, что вы обычно пишете в `wait`), то `wait` можно просто опустить.

Такой код

```testo
mouse click "password:".from_top(1).right_center().move_right(20); type "${password}"
mouse click "Continue".center_bottom()
wait "Restart now" timeout 10m; mouse click "Restart Now"
```

Будет полностью эквивалентен следующему коду

```testo
mouse click "password:".from_top(1).right_center().move_right(20); type "${password}"
mouse click "Continue".center_bottom()
mouse click "Restart Now" timeout 10m
```

Действие `mouse click` включает в себя ожидание нужной надписи в течение заданного таймаута и точно так же выдаст ошибку, как и `wait` в случае, если надпись так и не появится.

Обратите внимание, что, в отличие от установки Ubuntu Server, нам нельзя вытаскивать установочный диск на экране "Installation complete" (это приводит к зависанию виртуальной машины). Поэтому необходимо начать перезагруку и дождаться следующего экрана.

![Please remove](/static/tutorials/12_mouse/please_remove.png)

Для перезагрузки можно нажать Enter, а можно использовать комбинацию `stop, start`

```testo
mouse click "Restart Now" timeout 10m
wait "Please remove the installation medium" timeout 2m;
unplug dvd; stop; start
```

Наконец, необходимо выполнить логин в свежеустановленную ОС, чтобы убедиться, что всё хорошо. Экран логина выглядит так

![Login](/static/tutorials/12_mouse/login.png)

Чтобы понять, что появился именно логин-экран, можно было бы использовать `wait "${login}"`. Но мы выбрали логин `desktop`, который мог бы теоритически появляться на экране ещё и в процессе загрузки самой ОС. Может быть, это и не так, но мы не хотим лишний раз рисковать преждевременным срабатыванием действия `wait`, поэтому воспользуемся поисковым выражением `wait "${login} && "Not listed?"`. Это выражение сработает только в том случае, если на экране одновременно присуствует как `${login}`, так и `Not listed` - чего точно не возникнет во время загрузки ОС. Благодаря такому поисковому выражению мы можем быть достаточно уверены, что перед нами именно экран с логином.

В итоге завершающие действия будут выглядеть так:

```testo
unplug dvd; stop; start
wait "${login}" && "Not listed?" timeout 3m

mouse click "${login}";
wait "Password"; type "${password}"; mouse click "Sign In"
wait "Welcome to Ubuntu"
```

На этом установка Ubuntu Desktop, наконец, закончена, но нам нужно посмотреть ещё немного моментов

## Ещё примеры работы с мышкой

Для того, чтобы ещё немного поупражняться в работе с мышкой, мы попробуем написать тест, в котором мы создаем папку и затем перемещаем её в корзину.

Давайте приступим.

Для начала избавимся от этого назойливого экрана

![Welcome to Ubuntu](/static/tutorials/12_mouse/welcome_to_ubuntu.png)

```testo
test mouse_demo: install_ubuntu {
	ubuntu_desktop {
		mouse click "Welcome to Ubuntu"
		mouse click "Quit"
		abort "stop here"
	}
}
```

Обратите внимание, что мы не дожидаемся появления надписи `Quit`, а сразу же пытаемся на неё нажать. Напоминаю, что это возможно благодаря тому, что действие `mouse` сначала дожидается появления нужной надписи на экране (по умолчанию в течение 1 минуты).

Теперь попробуем создать новую папку.

```testo
test mouse_demo: install_ubuntu {
	ubuntu_desktop {
		mouse click "Welcome to Ubuntu"
		mouse click "Quit"

		mouse rclick 400 300
		mouse click "New Folder"
		wait "Folder name"; type "My folder"; mouse click "Create"
		wait "My folder" && !"Create"

		abort "stop here"
	}
}
```

Обратите внимание, что для создания папки мы должны кликнуть правой кнопкой мышки по пустому месту на рабочем столе. Для правого клика существует действие `mouse rclick`, а в качестве места для клика мы выбираем координаты "Примерно серердины экрана" (текущее разрешение у нас 800х600). В конце мы убеждаемся, что папка создалась (то есть на экране присутствует надпись `My folder`, но отсутствует надпись `Create`).

![Folder created](/static/tutorials/12_mouse/folder_created.png)

Теперь давайте попробуем удалить папку. Для этого мы сделаем следующее:

1. Наведем курсор на надпись `My folder`;
2. Зажмём левую кнопку мыши;
3. Переведём курсор на какое-нибудь место на рабочем столе
4. Переведём курсор на надпись `Trash`;
5. Отпустим левую кнопку мыши;
6. Убедимся, что надпись `My folder` пропала

Может вызывать вопрос пункт 3: зачем куда-то двигать папку перед тем, как положить её в корзину? На самом деле, по какой-то неизвестной причине графический интерфейс не отрабатывает, если пропустить этот шаг и попытаться перевести папку напрямую в корзину. Попробуйте и убедитесь в этом сами (можно в какой-то степени сказать, что мы нашли наш первый реальный баг и успешно воспроизвели его).

```testo
#Move the folder to trash
mouse move "My folder";
mouse hold lbtn
mouse move 200 300
mouse move "Trash"
mouse release

wait !"My folder"
```

Обратите внимание на новые действий `mouse hold` и `mouse release`. Действие `mouse hold` позволяет зажимать какую-то кнопку мышки, а `mouse release` отпускает все зажатые кнопки мышки.

> Нельзя зажимать одновременно несколько кнопок мыши.

> Если вы зажали какую-либо кнопку мыши, её **обязательно** необходимо отпустить в том же тесте, в котором вы ее зажали.

> Пока у вас в тесте зажата кнопка мыши, вы не можете выполнять никакие клики.

Наконец, давайте попробуем очистить корзину от только что переданной туда папки `My folder`

```testo
#Empty trash
mouse move 0 0 #move cursor aside
mouse dclick "Trash"
#Check if our folder actually is in the trash
wait "My folder"
mouse click "Empty"
mouse click "Empty Trash"
mouse move 0 0
wait "Trash is Empty"
```

Обратите внимание на действие `mouse dclick` - удобный способ выполнить двойной щелчок левой кнопки мышки. Также обратите внимание на `mouse move 0 0` - это действие нужно добавить, т.к. курсор загораживает надпись "Trash" после перемещения в корзину папки `My folder`, а также после очистки корзины курсор загораживает надпись `Trash is Empty`.  Все остальные действия должны быть достаточно понятны.

## Итоги

Управление мышкой в языке `testo-lang` реализовано в довольно мощном действии `mouse`. Формат действия выбран таким образом, чтобы простые клики выглядели максимально лаконично и не перегружнно, но при этом при надобности вы могли бы добавлять соответствующие уточнения максимально естественным образом. Попробуйте немного попрактиковаться в применении этого действия и скоро вы убедитесь, что в ней нет ничего сложного.

Готовые скрипты можно найти [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/12)