
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("000000") {
	stb::Image image("imgs/000000.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Все пакеты имеют последние версии").size() == 1);
}

TEST_CASE("000001") {
	stb::Image image("imgs/000001.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Введите пароль администратора").size() == 1);
}

TEST_CASE("000002") {
	stb::Image image("imgs/000002.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Type to search").size() == 1);
}

TEST_CASE("000003") {
	stb::Image image("imgs/000003.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Дополнительные настройки ОС").size() == 2);
	CHECK(tensor.match("Использовать по умолчанию ядро Hardened").size() == 1);
	CHECK(tensor.match("Включить блокировку консоли").size() == 1);
	CHECK(tensor.match("Включить блокировку интерпретаторов").size() == 1);
	CHECK(tensor.match("Использовать sudo с паролем").size() == 1);
	CHECK(tensor.match("Отключить автоматическую настройку сети").size() == 2);
	CHECK(tensor.match("Установить 32-х битный загрузчик").size() == 1);
	CHECK(tensor.match("Снимок экрана").size() == 1);
	CHECK(tensor.match("Справка").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000004") {
	stb::Image image("imgs/000004.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Веб-браузер").size() == 1);
	CHECK(tensor.match("Firefox").size() == 1);
	CHECK(tensor.match("Корзина").size() == 1);
	CHECK(tensor.match("Мой").size() == 1);
	CHECK(tensor.match("компьютер").size() == 1);
	CHECK(tensor.match("Помощь").size() == 1);
	CHECK(tensor.match("ASTRALINUX").size() == 1);
	CHECK(tensor.match("EN").size() == 1);
	CHECK(tensor.match("15:00").size() == 1);
	CHECK(tensor.match("Ср, 18 дек").size() == 1);
}

TEST_CASE("000005") {
	stb::Image image("imgs/000005.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Операционная система").size() == 1);
	CHECK(tensor.match("общего назначения").size() == 1);
	CHECK(tensor.match("Орёл").size() == 1);
	CHECK(tensor.match("Разметка дисков").size() == 1);
	CHECK(tensor.match("Выберите диск для разметки").size() == 1);
	CHECK(tensor.match("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000006") {
	stb::Image image("imgs/000006.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Операционная система").size() == 1);
	CHECK(tensor.match("общего назначения").size() == 1);
	CHECK(tensor.match("Орёл").size() == 1);
	CHECK(tensor.match("Разметка дисков").size() == 1);
	CHECK(tensor.match("Выберите диск для разметки").size() == 1);
	CHECK(tensor.match("SCSI1 (0,0,0) (sda) - 10.7 GB ATA QEMU HARDDISK").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000007") {
	stb::Image image("imgs/000007.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
	CHECK(tensor.match("Внимание!").size() == 1);
	CHECK(tensor.match("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
	CHECK(tensor.match("Нет").size() == 1);
	CHECK(tensor.match("Да").size() == 3);
	CHECK(tensor.match("Вернуться").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000008") {
	stb::Image image("imgs/000008.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Настройка учётных записей пользователей и паролей").size() == 1);
	CHECK(tensor.match("Неправильный пароль").size() == 1);
	CHECK(tensor.match("Введите пароль для нового администратора").size() == 1);
	CHECK(tensor.match("Введите пароль ещё раз").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000009") {
	stb::Image image("imgs/000009.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Настройка клавиатуры").size() == 1);
	CHECK(tensor.match("Способ переключения между национальной и латинской раскладкой").size() == 1);
	CHECK(tensor.match("Caps Lock").size() == 4);
	CHECK(tensor.match("правый Control").size() == 1);
	CHECK(tensor.match("Alt+Shift").size() == 3);
	CHECK(tensor.match("Control+Alt").size() == 1);
	CHECK(tensor.match("левая клавиша с логотипом").size() == 1);
	CHECK(tensor.match("без переключателя").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000010") {
	stb::Image image("imgs/000010.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Вход в astra").size() == 1);
	CHECK(tensor.match("среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
	CHECK(tensor.match("user").size() == 1);
	CHECK(tensor.match("Имя:").size() == 1);
	CHECK(tensor.match("Пароль:").size() == 1);
	CHECK(tensor.match("Тип сессии").size() == 1);
	CHECK(tensor.match("En").size() == 1);
}

TEST_CASE("000011") {
	stb::Image image("imgs/000011.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Вход в testo-astralinux").size() == 1);
	CHECK(tensor.match("четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
	CHECK(tensor.match("user").size() == 1);
	CHECK(tensor.match("Имя:").size() == 1);
	CHECK(tensor.match("Пароль:").size() == 1);
	CHECK(tensor.match("Тип сессии").size() == 1);
	CHECK(tensor.match("En").size() == 1);
}

TEST_CASE("000012") {
	stb::Image image("imgs/000012.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выбор программного обеспечения").size() == 1);
	CHECK(tensor.match("Выберите устанавливаемое программное обеспечение").size() == 1);
	CHECK(tensor.match("Базовые средства").size() == 1);
	CHECK(tensor.match("Рабочий стол Fly").size() == 1);
	CHECK(tensor.match("Приложения для работы с сенсорным экраном").size() == 1);
	CHECK(tensor.match("СУБД").size() == 1);
	CHECK(tensor.match("Средства удаленного доступа SSH").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000013") {
	stb::Image image("imgs/000013.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Настройка времени").size() == 1);
	CHECK(tensor.match("Выберите часовой пояс").size() == 1);
	CHECK(tensor.match("Москва").size() == 12);
	CHECK(tensor.match("Москва-01 - Калининград").size() == 1);
	CHECK(tensor.match("Москва+00 - Москва").size() == 1);
	CHECK(tensor.match("Москва+07 - Владивосток").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
}

TEST_CASE("000014") {
	stb::Image image("imgs/000014.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Дополнительные настройки ОС").size() == 2);
	CHECK(tensor.match("Использовать по умолчанию ядро Hardened").size() == 1);
	CHECK(tensor.match("Включить блокировку консоли").size() == 1);
	CHECK(tensor.match("Включить блокировку интерпретаторов").size() == 1);
	CHECK(tensor.match("Использовать sudo с паролем").size() == 1);
	CHECK(tensor.match("Отключить автоматическую настройку сети").size() == 2);
	CHECK(tensor.match("Установить 32-х битный загрузчик").size() == 1);
	CHECK(tensor.match("Справка").size() == 1);
	CHECK(tensor.match("Продолжить").size() == 1);
}

TEST_CASE("000015") {
	stb::Image image("imgs/000015.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Разметка дисков").size() == 1);
	CHECK(tensor.match("Выберите диск для разметки").size() == 1);
	CHECK(tensor.match("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
	CHECK(tensor.match("Вернуться").size() == 1);
}

TEST_CASE("000016") {
	stb::Image image("imgs/000016.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
	CHECK(tensor.match("Внимание!").size() == 1);
	CHECK(tensor.match("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
	CHECK(tensor.match("Нет").size() == 1);
	CHECK(tensor.match("Да").size() == 3);
}

TEST_CASE("000017") {
	stb::Image image("imgs/000017.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("astra login:").size() == 2);
	CHECK(tensor.match("Password:").size() == 2);
	CHECK(tensor.match("Login incorrect").size() == 1);
	CHECK(tensor.match("You have mail.").size() == 1);
	CHECK(tensor.match("user@astra:~$").size() == 1);
}

TEST_CASE("000018") {
	stb::Image image("imgs/000018.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Графическая установка").size() == 1);
	CHECK(tensor.match("Установка").size() == 2);
	CHECK(tensor.match("Режим восстановления").size() == 1);
	CHECK(tensor.match("Орёл - город воинской славы").size() == 1);
	CHECK(tensor.match("Русский").size() == 1);
	CHECK(tensor.match("English").size() == 1);
	CHECK(tensor.match("F1 Язык").size() == 1);
	CHECK(tensor.match("F2 Параметры").size() == 1);
}

TEST_CASE("000019") {
	stb::Image image("imgs/000019.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("ROOT PASSWORD").size() == 1);
	CHECK(tensor.match("Root password").size() == 1);
}

TEST_CASE("000020") {
	stb::Image image("imgs/000020.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Install Ubuntu Server").size() == 2);
	CHECK(tensor.match("Install Ubuntu Server with the HWE kernel").size() == 1);
	CHECK(tensor.match("Test memory").size() == 1);
	CHECK(tensor.match("Boot from first hard disk").size() == 1);
}

TEST_CASE("000021") {
	stb::Image image("imgs/000021.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Select a language").size() == 1);
}

TEST_CASE("000022") {
	stb::Image image("imgs/000022.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Select your location").size() == 1);
}

TEST_CASE("000023") {
	stb::Image image("imgs/000023.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Language").size() == 2);
	CHECK(tensor.match("English").size() == 1);
	CHECK(tensor.match("Русский").size() == 1);
	CHECK(tensor.match("F1 Help").size() == 1);
	CHECK(tensor.match("F2 Language").size() == 1);
	CHECK(tensor.match("F3 Keymap").size() == 1);
	CHECK(tensor.match("F4 Modes").size() == 1);
	CHECK(tensor.match("F5 Accessibility").size() == 1);
	CHECK(tensor.match("F6 Other Options").size() == 1);
}

TEST_CASE("000024") {
	stb::Image image("imgs/000024.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Install Ubuntu Server").size() == 2);
	CHECK(tensor.match("Install Ubuntu Server with the HWE kernel").size() == 1);
	CHECK(tensor.match("Check disc for defects").size() == 1);
	CHECK(tensor.match("Test memory").size() == 1);
	CHECK(tensor.match("Boot from first hard disk").size() == 1);
}

TEST_CASE("000025") {
	stb::Image image("imgs/000025.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Welcome!").size() == 1);
	CHECK(tensor.match("Добро пожаловать!").size() == 1);
	CHECK(tensor.match("Please choose your preferred language.").size() == 1);
	CHECK(tensor.match("English").size() == 1);
	CHECK(tensor.match("Русский").size() == 1);
	CHECK(tensor.match("Use UP, DOWN and ENTER keys to select your language.").size() == 1);
}

TEST_CASE("000026") {
	stb::Image image("imgs/000026.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Install Ubuntu").size() == 1);
}

TEST_CASE("000027") {
	stb::Image image("imgs/000027.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Network connections").size() == 1);
	CHECK(tensor.match("192.168.122.219/24").size() == 1);
	CHECK(tensor.match("52:54:00:45:12:e7").size() == 1);
}

TEST_CASE("000028") {
	stb::Image image("imgs/000028.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Configure proxy").size() == 1);
	CHECK(tensor.match("\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
}

TEST_CASE("000029") {
	stb::Image image("imgs/000029.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Installation complete!").size() == 1);
	CHECK(tensor.match("Reboot Now").size() == 1);
}

TEST_CASE("000030") {
	stb::Image image("imgs/000030.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Please remove the installation medium, then press ENTER:").size() == 1);
}

TEST_CASE("000031") {
	stb::Image image("imgs/000031.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Setting up daemon").size() == 1);
	CHECK(tensor.match("Unpacking python").size() == 2);
	CHECK(tensor.match("Unpacking python2.7").size() == 1);
	CHECK(tensor.match("Selecting previously unselected package").size() == 3);
	CHECK(tensor.match("root@client:/home/user#").size() == 1);
}

TEST_CASE("000032") {
	stb::Image image("imgs/000032.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Продолжить на выбранном языке?").size() == 1);
	CHECK(tensor.match("Да").size() == 1);
	CHECK(tensor.match("Русский").size() == 1);
	CHECK(tensor.match("English (United States)").size() == 1);
	CHECK(tensor.match("Добро пожаловать").size() == 1);
}

TEST_CASE("000033") {
	stb::Image image("imgs/000033.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Давайте начнем с региона. Это правильно?").size() == 1);
	CHECK(tensor.match("Парагвай").size() == 1);
	CHECK(tensor.match("Перу").size() == 1);
	CHECK(tensor.match("Польша").size() == 1);
	CHECK(tensor.match("Португалия").size() == 1);
	CHECK(tensor.match("Пуэрто-Рико").size() == 1);
	CHECK(tensor.match("Реюньон").size() == 1);
	CHECK(tensor.match("Россия").size() == 1);
	CHECK(tensor.match("Да").size() == 2);
	CHECK(tensor.match("Основы").size() == 1);
}

TEST_CASE("000034") {
	stb::Image image("imgs/000034.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Лицензионное соглашение о Windows 10").size() == 1);
}

TEST_CASE("000035") {
	stb::Image image("imgs/000035.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Если вы подключитесь к Интернету").size() == 1);
}

TEST_CASE("000036") {
	stb::Image image("imgs/000036.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Войдите с помощью учетной записи").size() == 1);
}

TEST_CASE("000037") {
	stb::Image image("imgs/000037.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Кто будет использовать этот компьютер").size() == 1);
}

TEST_CASE("000038") {
	stb::Image image("imgs/000038.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Удобная работа на разных устройствах").size() == 1);
}

TEST_CASE("000039") {
	stb::Image image("imgs/000039.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Программа конфигурации Валидата CSP").size() == 1);
	CHECK(tensor.match("ДСЧ").size() == 1);
	CHECK(tensor.match("Считыватели ключа").size() == 1);
	CHECK(tensor.match("Ключи").size() == 1);
	CHECK(tensor.match("Сертификаты").size() == 1);
	CHECK(tensor.match("Совместимость").size() == 1);
	CHECK(tensor.match("Сервис").size() == 1);
	CHECK(tensor.match("Выберите считыватель ключа").size() == 1);
	CHECK(tensor.match("Считыватель съёмного диска").size() == 1);
	CHECK(tensor.match("Считыватель 'Соболь'").size() == 1);
	CHECK(tensor.match("Считыватель 'Аккорд'").size() == 1);
	CHECK(tensor.match("Считыватель реестра").size() == 1);
	CHECK(tensor.match("Считыватель ruToken").size() == 1);
	CHECK(tensor.match("Считыватель eToken").size() == 1);
	CHECK(tensor.match("Считыватель vdToken").size() == 2);
	CHECK(tensor.match("Отмена").size() == 2);
	CHECK(tensor.match("Применить").size() == 1);
	CHECK(tensor.match("Validata").size() == 1);
}

TEST_CASE("000040") {
	stb::Image image("imgs/000040.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Настройка Windows").size() == 1);
	CHECK(tensor.match("Windows завершает применение параметров").size() == 1);
	CHECK(tensor.match("Справка").size() == 1);
	CHECK(tensor.match("Русский (Россия)").size() == 1);
}

TEST_CASE("000041") {
	stb::Image image("imgs/000041.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите имя пользователя для вашей учетной записи, а также имя компьютера в сети.").size() == 1);
	CHECK(tensor.match("Введите имя пользователя (например, Андрей):").size() == 1);
	CHECK(tensor.match("Введите имя компьютера:").size() == 1);
	CHECK(tensor.match("Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000042") {
	stb::Image image("imgs/000042.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Введите ключ продукта Windows").size() == 1);
	CHECK(tensor.match("ключ продукта").size() == 2);
	CHECK(tensor.match("КЛЮЧ ПРОДУКТА").size() == 1);
	CHECK(tensor.match("Наклейка с ключом продукта выглядит так:").size() == 1);
	CHECK(tensor.match("XXXXX-XXXXX-XXXXX-XXXXX-XXXXX").size() == 1);
	CHECK(tensor.match("Автоматически активировать Windows при подключении к Интернету").size() == 1);
	CHECK(tensor.match("Что такое активация?").size() == 1);
	CHECK(tensor.match("Заявление о конфиденциальности").size() == 1);
	CHECK(tensor.match("Пропустить").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000043") {
	stb::Image image("imgs/000043.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установите пароль для своей учетной записи").size() == 1);
	CHECK(tensor.match("Введите пароль (рекомендуется):").size() == 1);
	CHECK(tensor.match("Подтверждение пароля:").size() == 1);
	CHECK(tensor.match("Введите подсказку для пароля:").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000044") {
	stb::Image image("imgs/000044.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите раздел для установки Windows").size() == 1);
	CHECK(tensor.match("Файл").size() == 1);
	CHECK(tensor.match("Полный раз...").size() == 1);
	CHECK(tensor.match("Свободно").size() == 1);
	CHECK(tensor.match("Тип").size() == 1);
	CHECK(tensor.match("Незанятое место на диске 0").size() == 1);
	CHECK(tensor.match("20.0 ГБ").size() == 2);
	CHECK(tensor.match("Обновить").size() == 1);
	CHECK(tensor.match("Настройка диска").size() == 1);
	CHECK(tensor.match("Загрузка").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
	CHECK(tensor.match("Сбор информации").size() == 1);
	CHECK(tensor.match("Установка Windows").size() == 2);
}

TEST_CASE("000045") {
	stb::Image image("imgs/000045.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите тип установки").size() == 1);
	CHECK(tensor.match("Обновление").size() == 3);
	CHECK(tensor.match("Полная установка (дополнительные параметры)").size() == 1);
	CHECK(tensor.match("Помощь в принятии решения").size() == 1);
}

TEST_CASE("000046") {
	stb::Image image("imgs/000046.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите операционную систему, которую следует установить").size() == 1);
	CHECK(tensor.match("Операционная система").size() == 1);
	CHECK(tensor.match("Архитектура").size() == 1);
	CHECK(tensor.match("Дата").size() == 1);
	CHECK(tensor.match("Windows 7 Начальная").size() == 2);
	CHECK(tensor.match("Windows 7 Домашняя базовая").size() == 2);
	CHECK(tensor.match("Windows 7 Домашняя расширенная").size() == 2);
	CHECK(tensor.match("Windows 7 Профессиональная").size() == 2);
	CHECK(tensor.match("Windows 7 Максимальная").size() == 2);
	CHECK(tensor.match("x86").size() == 5);
	CHECK(tensor.match("x64").size() == 4);
	CHECK(tensor.match("11/20/2010").size() == 5);
	CHECK(tensor.match("11/21/2010").size() == 4);
	CHECK(tensor.match("Описание:").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000047") {
	stb::Image image("imgs/000047.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("My language is English").size() == 1);
	CHECK(tensor.match("Мой язык - русский").size() == 1);
}

TEST_CASE("000048") {
	stb::Image image("imgs/000048.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Устанавливаемый язык:").size() == 1);
	CHECK(tensor.match("Формат времени и денежных единиц:").size() == 1);
	CHECK(tensor.match("Раскладка клавиатуры или метод ввода:").size() == 1);
	CHECK(tensor.match("Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
}

TEST_CASE("000049") {
	stb::Image image("imgs/000049.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Для продолжения требуется перезагрузка Windows").size() == 1);
	CHECK(tensor.match("Перезагрузка через 5 сек.").size() == 1);
	CHECK(tensor.match("Перезагрузить сейчас").size() == 1);
}

TEST_CASE("000050") {
	stb::Image image("imgs/000050.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Проверьте настройку даты и времени").size() == 1);
	CHECK(tensor.match("Часовой пояс:").size() == 1);
	CHECK(tensor.match("Автоматический переход на летнее время и обратно").size() == 1);
	CHECK(tensor.match("Дата:").size() == 1);
	CHECK(tensor.match("Время:").size() == 1);
	CHECK(tensor.match("Ноябрь 2019").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000051") {
	stb::Image image("imgs/000051.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите текущее место расположения компьютера").size() == 1);
	CHECK(tensor.match("Домашняя сеть").size() == 1);
	CHECK(tensor.match("Рабочая сеть").size() == 1);
	CHECK(tensor.match("Общественная сеть").size() == 1);
	CHECK(tensor.match("Если не уверены, выбирайте общественную сеть.").size() == 1);
}

TEST_CASE("000052") {
	stb::Image image("imgs/000052.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Ознакомьтесь с условиями лицензии").size() == 1);
	CHECK(tensor.match("УСЛОВИЯ ЛИЦЕНЗИИ НА ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ MICROSOFT").size() == 1);
	CHECK(tensor.match("WINDOWS 7 МАКСИМАЛЬНАЯ С ПАКЕТОМ ОБНОВЛЕНИЯ 1").size() == 1);
	CHECK(tensor.match("Я принимаю условия лицензии").size() == 1);
	CHECK(tensor.match("Далее").size() == 1);
}

TEST_CASE("000053") {
	stb::Image image("imgs/000053.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Помогите автоматически защитить компьютер и улучшить Windows").size() == 1);
	CHECK(tensor.match("Использовать рекомендуемые параметры").size() == 1);
	CHECK(tensor.match("Устанавливать только наиболее важные обновления").size() == 1);
	CHECK(tensor.match("Отложить решение").size() == 1);
	CHECK(tensor.match("Подробнее об этих параметрах").size() == 1);
	CHECK(tensor.match("Заявление о конфиденциальности").size() == 1);
}

TEST_CASE("000054") {
	stb::Image image("imgs/000054.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установка Windows").size() == 1);
	CHECK(tensor.match("Windows 7").size() == 1);
	CHECK(tensor.match("Установить").size() == 1);
	CHECK(tensor.match("Что следует знать перед выполнением установки Windows").size() == 1);
	CHECK(tensor.match("Восстановление системы").size() == 1);
}

TEST_CASE("000055") {
	stb::Image image("imgs/000055.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Корзина").size() == 1);
	CHECK(tensor.match("RU").size() == 1);
	CHECK(tensor.match("13:56").size() == 1);
	CHECK(tensor.match("23.11.2019").size() == 1);
}

TEST_CASE("000056") {
	stb::Image image("imgs/000056.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установка Windows...").size() == 1);
	CHECK(tensor.match("Копирование файлов Windows").size() == 1);
	CHECK(tensor.match("Распаковка файлов Windows (0%)").size() == 1);
	CHECK(tensor.match("Установка компонентов").size() == 1);
	CHECK(tensor.match("Установка обновлений").size() == 1);
	CHECK(tensor.match("Завершение установки").size() == 1);
}

TEST_CASE("000057") {
	stb::Image image("imgs/000057.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("RU").size() == 1);
	CHECK(tensor.match("Петя").size() == 1);
	CHECK(tensor.match("Пароль").size() == 1);
	CHECK(tensor.match("Windows 7 Максимальная").size() == 1);
}

TEST_CASE("000058") {
	stb::Image image("imgs/000058.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Для создание шаров-мишеней").size() == 1);
	CHECK(tensor.match("нажимайте левой кнопкой мыши").size() == 1);
	CHECK(tensor.match("пока индикатор не заполнится").size() == 1);
	CHECK(tensor.match("целиком").size() == 1);
	CHECK(tensor.match("Корзина").size() == 1);
	CHECK(tensor.match("Континент").size() == 1);
	CHECK(tensor.match("TLS-клиент").size() == 1);
}

TEST_CASE("000059") {
	stb::Image image("imgs/000059.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Нажмите Enter чтобы провести инсталляцию вручную").size() == 1);
}

TEST_CASE("000060") {
	stb::Image image("imgs/000060.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("bound to 192.168.122.145 -- renewal in 1720 seconds.").size() == 1);
	CHECK(tensor.match("Отсутствует электронный замок Соболь").size() == 1);
	CHECK(tensor.match("Дальнейшая работа будет производиться без него").size() == 1);
	CHECK(tensor.match("продолжить? (y/n):").size() == 1);
}

TEST_CASE("000061") {
	stb::Image image("imgs/000061.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите вариант установки:").size() == 1);
	CHECK(tensor.match("4: ЦУС с сервером доступа (отладочная версия)").size() == 1);
	CHECK(tensor.match("Введите номер варианта [1..7]:").size() == 1);
	CHECK(tensor.match("Выберите действие").size() == 1);
	CHECK(tensor.match("1: Установка").size() == 1);
	CHECK(tensor.match("Введите номер варианта [1..3]:").size() == 1);
	CHECK(tensor.match("Установка << Континент >>").size() == 1);
	CHECK(tensor.match("продолжить? (y/n):").size() == 2);
}

TEST_CASE("000062") {
	stb::Image image("imgs/000062.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Обнаруженные интерфейсы:").size() == 2);
	CHECK(tensor.match("Укажите номер внешнего интерфейса:").size() == 1);
	CHECK(tensor.match("Введите внешний IP адрес шлюза:").size() == 1);
	CHECK(tensor.match("Продолжить? (Y/N):").size() == 3);
	CHECK(tensor.match("Укажите номер внутреннего интерфейса.").size() == 1);
	CHECK(tensor.match("Введите внутренний IP адрес шлюза:").size() == 1);
	CHECK(tensor.match("Введите адрес маршрутизатора по умолчанию:").size() == 1);
	CHECK(tensor.match("Использовать внешний носитель для инициализации? (Y/N):").size() == 1);
}

TEST_CASE("000063") {
	stb::Image image("imgs/000063.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Вставьте носитель для записи ключа администратора ЦУС и нажмите Enter").size() == 1);
	CHECK(tensor.match("Ключи администратора успешно сохранены").size() == 1);
	CHECK(tensor.match("Создать учетную запись локального администратора? (Y/N):").size() == 1);
	CHECK(tensor.match("Введите логин администратора:").size() == 1);
	CHECK(tensor.match("Введите пароль:").size() == 1);
	CHECK(tensor.match("Повторите пароль:").size() == 2);
	CHECK(tensor.match("Конфигурация ЦУС завершена").size() == 1);
	CHECK(tensor.match("1: Завершение работы").size() == 1);
	CHECK(tensor.match("2: Перезагрузка").size() == 1);
	CHECK(tensor.match("3: Управление конфигурацией").size() == 1);
	CHECK(tensor.match("4: Настройка безопасности").size() == 1);
	CHECK(tensor.match("5: Настройка СД").size() == 1);
	CHECK(tensor.match("6: Тестирование").size() == 1);
	CHECK(tensor.match("0: Выход").size() == 1);
	CHECK(tensor.match("Выберите пункт меню (0 - 6):").size() == 1);
}

TEST_CASE("000064") {
	stb::Image image("imgs/000064.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Please select Language").size() == 1);
	CHECK(tensor.match("English").size() == 1);
	CHECK(tensor.match("Russian").size() == 1);
	CHECK(tensor.match("Autodeploy").size() == 1);
	CHECK(tensor.match("Automatic boot in 30 seconds").size() == 1);
}

TEST_CASE("000065") {
	stb::Image image("imgs/000065.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установить Континент 4.1.0.919").size() == 1);
	CHECK(tensor.match("Тест памяти").size() == 1);
}

TEST_CASE("000066") {
	stb::Image image("imgs/000066.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выберите тип платформы").size() == 1);
	CHECK(tensor.match("Настраиваемая").size() == 1);
}

TEST_CASE("000067") {
	stb::Image image("imgs/000067.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Установка").size() == 1);
	CHECK(tensor.match("Введите идентификатор шлюза").size() == 1);
}

TEST_CASE("000068") {
	stb::Image image("imgs/000068.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Выбор метода аутентификации").size() == 1);
	CHECK(tensor.match("Соболь").size() == 1);
	CHECK(tensor.match("Учётная запись/пароль").size() == 1);
	CHECK(tensor.match("Отмена").size() == 1);
}

TEST_CASE("000069") {
	stb::Image image("imgs/000069.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Главное меню").size() == 1);
	CHECK(tensor.match("Сведения").size() == 1);
	CHECK(tensor.match("Сведения").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
	CHECK(tensor.match("Сведения").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
	CHECK(tensor.match("Инициализация").size() == 1);
	CHECK(tensor.match("Инициализация").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
	CHECK(tensor.match("Инициализация").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
	CHECK(tensor.match("Журналы").size() == 1);
	CHECK(tensor.match("Инструменты").size() == 1);
	CHECK(tensor.match("Настройки").size() == 1);
	CHECK(tensor.match("Change Language/Сменить язык").size() == 1);
	CHECK(tensor.match("Завершение работы устройства").size() == 1);
}

TEST_CASE("000070") {
	stb::Image image("imgs/000070.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Повторная инициализация").size() == 1);
	CHECK(tensor.match("Сертификаты").size() == 1);
	CHECK(tensor.match("Настройка ЦУС").size() == 1);
}

TEST_CASE("000071") {
	stb::Image image("imgs/000071.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Успешно.").size() == 1);
	CHECK(tensor.match("Нажмите Enter").size() == 1);
}

TEST_CASE("000072") {
	stb::Image image("imgs/000072.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Инициализировать устройство как:").size() == 1);
	CHECK(tensor.match("[ Начать инициализацию ]").size() == 1);
	CHECK(tensor.match("[ Начать инициализацию ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
	CHECK(tensor.match("[ Начать инициализацию ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
	CHECK(tensor.match("[ Отмена ]").size() == 1);
	CHECK(tensor.match("[ Отмена ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
	CHECK(tensor.match("[ Отмена ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
}

TEST_CASE("000073") {
	stb::Image image("imgs/000073.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Инструменты").size() == 1);
	CHECK(tensor.match("Экспорт конфигурации УБ на носитель").size() == 1);
}

TEST_CASE("000074") {
	stb::Image image("imgs/000074.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Меню настроек").size() == 1);
	CHECK(tensor.match("Применение локальной политики").size() == 1);
}

TEST_CASE("000075") {
	stb::Image image("imgs/000075.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Сертификаты УЦ").size() == 1);
}

TEST_CASE("000076") {
	stb::Image image("imgs/000076.png");
	nn::Tensor tensor = nn::find_text(&image);
	CHECK(tensor.match("Очистить локальные журналы?").size() == 1);
	CHECK(tensor.match("Да").size() == 1);
	CHECK(tensor.match("Да").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
	CHECK(tensor.match("Да").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
	CHECK(tensor.match("Нет").size() == 1);
	CHECK(tensor.match("Нет").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
	CHECK(tensor.match("Нет").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
}
