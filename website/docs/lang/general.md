# Общая структура скриптовых файлов

На верхнем уровне скриптовый язык Testo можно рассматривать как
последовательность объявлений. Помимо объявлений, на верхнем уровне
можно использовать только директиву `include`.

Объявления делятся на:

-   `machine` - объявление виртуальной машины
-   `flash` - объявление виртуального флеш-накопителя
-   `network` - объявление виртуальной сети
-   `param` - объявление параметра
-   `test` - объявление теста
-   `macro` - объявление макроса

Все эти объявления могут быть расположены в любом порядке и могут быть
распределены по нескольким файлам. Однако, виртуальные машины,
флеш-накопители, параметры и макросы должны быть объявлены до того, как
к ним произойдет обращение. Чтобы включить в скриптовый файл все
объявления из другого файла, можно использовать директиву `include`.
Директива `include` может располагаться только между другими
объявлениями, ее запрещено использовать внутри объявлений.

Архитектура написания тестов выглядит следующим образом. В объявлениях
`machine`, `flash` и `network` разработчик перечисляет, какие ресурсы
ему понадобятся в тестовом сценарии, описывая их атрибуты. С помощью
директивы `param` разработчик перечисляет глобальные константы, которые
будут использоваться в дальнейшем в тексте тестовых сценариев. Объявив
нужные ресурсы и константы, разработчик может переходить к описанию
тестов.

Тесты состоят из набора команд. Каждая команда состоит из одной или
нескольких виртуальных машин и действия (или набора действий),
применяемых к этой машине. Если одно из действий не будет выполнено
успешно, то тест считается ошибочным и управление переходит к следующему
запланированному тесту (если не указан аргумент командной строки
`--stop_on_fail`).

Наиболее часто повторяемые действия могут быть оформлены в виде макросов
с помощью директивы `macro`. Макросы могут принимать параметры.