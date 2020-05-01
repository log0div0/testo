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
	disk_size: 20Gb
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

Сразу после 

![Write changes](/static/tutorials/12_mouse/write_changes_to_disk.png)
