# Макросы

Макросы позволяют объединять часто используемые комбинации действий в
виде обособленных именованных блоков. Формат объявления макроса выглядит
следующим образом:

```text
macro <name> ([arg1, arg2, ... argn="default_value1", argn+1="default_value2" ...]) {
	<attr1>: <value1>
	<attr2>: <value1>
	<attr3>: <value1>
	...
}
```

Макросы должны иметь имя в формате идентификатора, уникальное среди
макросов. Макросы могут принимать агументы, к которым затем можно
получать доступ внутри тела макроса. На текущий момент можно передавать
только строковые аргументы. Аргументы могут иметь значение по умолчанию. Внутри значения по умолчанию можно обращаться к параметрам как это
показано [здесь](var_refs).

В теле макроса можно использовать все те конструкции, которые можно
указывать в теле команд, из которых состоит тест: действия, условия,
циклы, вызов других макросов.

Формат вызова макроса можно посмотреть
[здесь](actions#вызов-макроса)