
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Astra Linux Orel/Установка (GUI)") {
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/additional_options.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/desktop.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2-alt.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/grub.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(tensor.match("Внимание!").size() == 1);
		CHECK(tensor.match("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(tensor.match("Нет").size() == 1);
		CHECK(tensor.match("Да").size() == 3);
		CHECK(tensor.match("Вернуться").size() == 1);
		CHECK(tensor.match("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/incorrect_password.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Настройка учётных записей пользователей и паролей").size() == 1);
		CHECK(tensor.match("Неправильный пароль").size() == 1);
		CHECK(tensor.match("Введите пароль для нового администратора").size() == 1);
		CHECK(tensor.match("Введите пароль ещё раз").size() == 1);
		CHECK(tensor.match("Вернуться").size() == 1);
		CHECK(tensor.match("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/keyboard.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Вход в astra").size() == 1);
		CHECK(tensor.match("среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
		CHECK(tensor.match("user").size() == 1);
		CHECK(tensor.match("Имя:").size() == 1);
		CHECK(tensor.match("Пароль:").size() == 1);
		CHECK(tensor.match("Тип сессии").size() == 1);
		CHECK(tensor.match("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen_alt.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Вход в testo-astralinux").size() == 1);
		CHECK(tensor.match("четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
		CHECK(tensor.match("user").size() == 1);
		CHECK(tensor.match("Имя:").size() == 1);
		CHECK(tensor.match("Пароль:").size() == 1);
		CHECK(tensor.match("Тип сессии").size() == 1);
		CHECK(tensor.match("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/programs.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/time.png");
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
}

TEST_CASE("Astra Linux Orel/Установка (Консоль)") {
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/additional_options.png");
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
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/disks-2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Разметка дисков").size() == 1);
		CHECK(tensor.match("Выберите диск для разметки").size() == 1);
		CHECK(tensor.match("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(tensor.match("Вернуться").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/grub.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(tensor.match("Внимание!").size() == 1);
		CHECK(tensor.match("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(tensor.match("Нет").size() == 1);
		CHECK(tensor.match("Да").size() == 3);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/login.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("astra login:").size() == 2);
		CHECK(tensor.match("Password:").size() == 2);
		CHECK(tensor.match("Login incorrect").size() == 1);
		CHECK(tensor.match("You have mail.").size() == 1);
		CHECK(tensor.match("user@astra:~$").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/Start.png");
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
}
