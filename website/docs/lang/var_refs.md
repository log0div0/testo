# Обращение к параметрам

В языке Testo существует особенность обобщать процесс тестирования с
помощью параметров. На текущий момент параметры могут принимать только
текстовые значения. Существует четыре способа задания параметров:

- Через глобальную константу в тексте файла .testo
- Через глобальную константу, переданную в качестве аргумента
  командой строки
- Через счетчик в цикле `for`
- В качестве аргументов в макросах

Обращаться к параметрам можно в следующих местах скриптов:

- в составе [действий](actions),
  которые в качестве аргумента принимают строки
- в конфигурациях виртуальной машины или виртуального
  флеш-накопителя (в тех атрибутах, которые могут принимать
  строковые значения). Обращение к параметру во всех случаях
  выглядит одинаково:

```testo
"${VAR_REF}"
```

Где `VAR_REF` - это идентификатор с именем параметра

Обращаться к параметру можно прямо внутри строки, не прерывая ее:

```testo
"Hello ${VAR_REF}"
```

Обращаться к параметрам можно и в многострочных строках и в
select-выражениях

```testo
"""Hello
	${VAR_REF}
"""

js """
	find_text("Menu entry").foreground("${foreground_colour}).background("${gray}")
"""
```

## Экранирование обращения к параметрам

Для экранирования сочетания символов `${` необходимо добавить еще один
знак доллара в начале: `$${`.

## Порядок разрешения значения параметров

При обращении к параметру его значение вычисляется по следующему
алгоритму, причем если на каком-то шаге находится значение, то алгоритм
прекращается:

1)  Если обращение к параметру происходит внутри цикла `for`, то
    проверяется, является ли параметр счетчиком цикла
2)  Если обращение к параметру происходит внутри макроса, то
    проверяется, является ли параметр аргументом макроса
3)  Если на предыдущих этапах не получилось вычислить значение, то
    проверяется, является ли параметр глобальной константой
4)  Возвращается пустая строка