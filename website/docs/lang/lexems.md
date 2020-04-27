# Базовые конструкции языка

## Литералы

В языке Testo существуют следующие типы литералов:

-   Числовые литералы
-   Строковые литералы без переноса строк
-   Строковые литералы с переносом строк
-   Логические литералы
-   Спецификатор количества памяти
-   Спецификатор количества времени

### Числовые литералы

Числовые литералы представляют собой непрерывную последовательность
цифр. Перед цифрами может идти знак \"+\" или \"-\"

### Строковые литералы

**Строковые литералы без переноса строк** представляют собой любую
последовательность символов, заключенную между двойными кавычками.
Например, `"Hello world!"`. Внутри таких строк можно исопльзовать
перенос строки, предварительно поставив экранирующий символ. Например,

```testo
"This is example \
of some \
multiline string"
```

**Строковые литералы с переносом строк** представляют собой любую
последовательность символов, заключенную между тройными двойными
кавычками `"""`. Внутри таких строк можно использовать перенос строки.
Например,

```testo
"""This is example
of some
multiline string
"""
```

Внутри строк можно использовать экранирующий символ `\`. Например,
строка `This is \"Hello World\" example` транслируется в
`This is "Hello World" example`

В многострочных литералах нет необходимости экранировать двойные кавычки
`"`. Необходимо экранировать лишь тройные кавычки, если они встречаются
в тексте `"""`

```testo
"""This is example
"Hello world"
of multiline string
"""

"""This is example
\"""Triple qouted part\"""
of multiline string
"""
```

Внутри строк можно использовать обращение к переменным

Например, имеет место следующая конкатенация строк

```testo
"""This is example of ${VAR_REF}
string concatenation
"""
```

В этом случае конечное значение строки будет зависеть от значения
переменной `VAR_REF`

### Логические литералы

**Логические литералы** - это зарезервированные идентификаторы `true` и
`false`. В настоящее время логические литералы используются только в
качестве значений для некоторых атрибутов.

### Спецификатор количества памяти

**Спецификатор количества памяти** имеет формат
`Число + размерность памяти`. Размерность памяти может принимать
значения `Mb`, `Kb` и `Gb`. Примеры: `512Mb`, `3Gb`, `640Kb`

### Спецификатор количества времени

**Спецификатор количества времени** имеет формат
`Число + размерность временного отрезка`. Размерность временного отрезка
может принимать значения `ms` (миллисекунды), `s` (секунды), `m`
(минуты) и `h` (часы). Примеры: `600s`, `1m`, `5h`, `50ms`

## Идентификаторы

Для обозначения имен виртуальных машин, флеш-накопителей, тестов и
других сущностей используются идентификаторы. Идентификатор должен
начинаться с буквы английского алфавита или знака подчеркивания. Второй
и последующий символ могут быть любой буквой английского алфавита,
цифрой, знаком подчеркивания или дефисом.

Примеры: `example`, `another_example`, `_this_is_good_too`,
`And_even-this233-`

Неправильные идентификаторы: `example with spaces`, `5example`

## Ключевые слова

Некоторые идентификаторы зарезервированы как ключевые слова.
Использовать их для наименования сущностей нельзя.

- `abort` - Действие \"прекратить тест\"
- `print` - Действие \"вывести сообщение на экран\"
- `type` - Действие \"напечатать строку на клавиатуре\"
- `wait` - Действие \"дождаться отображения строки на экране\"
- `sleep` - Действие \"безусловное ожидание\"
- `mouse` - Действие, связанное с мышкой
- `move` - Действие \"передвинуть курсор мышки\"
- `click` - Действие \"нажать левую кнопку мышки\"
- `lclick` - То же самое, что и `click`
- `rclick` - Действие \"нажать правую кнопку мышки\"
- `dclick` - Действие \"дважды нажать левую кнопку мышки\"
- `hold` - Действие \"зажать кнопку мышки\"
- `release` - Действие \"отпустить кнопку мышки\"
- `lbtn` - Спецификатор левой кнопки мышки в действии `hold`
- `rbtn` - Спецификатор правой кнопки мышки в действии `hold`
- `check` - Проверка \"проверить наличие строки на экране\"
- `press` - Действие \"нажать клавишу\"
- `plug` - Действие \"подключить\"
- `unplug` - Действие \"отключить\"
- `start` - Действие \"включить питание\"
- `stop` - Действие \"отключить питание\"
- `shutdown` - Действие \"нажать на кнопку выключения питания\"
- `exec` - Действие \"выполнить команду на виртуальной машине\"
- `copyto` - Действие \"скопировать файлы на виртуальную машину\"
- `copyfrom` - Действие \"скопировать файлы из виртуальной машины\"
- `timeout` - Указание таймаута для некоторых действий
- `interval` - Указание временного интервала между итерациями внутри
  некоторых команд
- `test` - Начало объявления теста
- `machine` - Начало объявления виртуальной машины
- `flash` - Начало объявления виртуального флеш-накопителя
- `network` - Начало объявления витруальной сети
- `param` - Начало объявления параметра (глобальной константы)
- `macro` - Начало объявления макроса
- `dvd` - Спецификатор dvd-привода в действии `plug` и `unplug`
- `if` - Начало условия
- `else` - Начало действий в случае, если условие в `if` не
  сработало
- `for` - Начало цикла
- `IN` - Указывается перед диапазоном в циклах
- `RANGE` - Указывается в начале объявления диапазона
- `break` - Действие \"выйти из цикла\"
- `continue` - Действие \"перейти к следующей итерации в цикле\"
- `include` - Директива к включению другого файла с тестовыми
  сценариями
- `js` - начало объявления javascript-скрипта
- `LESS` - Проверка на то, что одно число меньше другого
- `GREATER` - Проверка на то, что одно число больше другого
- `EQUAL` - Проверка на равенство двух чисел
- `STRLESS` - Проверка на то, что одна строка меньше другой
- `STRGREATER` - Проверка на то, что одна строка больше другой
- `STREQUAL` - Проверка на равенство двух строк
- `NOT` - Отрицание значения выражения
- `AND` - Логическое \"И\" значений двух выражений
- `OR` - Логическое \"ИЛИ\" значений двух выражений
- `true` - Логическая единица
- `false` - Логический ноль