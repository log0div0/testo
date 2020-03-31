
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Astra Linux Orel/Установка (GUI)") {
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/additional_options.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Дополнительные настройки ОС").size() == 2);
		CHECK(ocr.search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(ocr.search("Включить блокировку консоли").size() == 1);
		CHECK(ocr.search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(ocr.search("Использовать sudo с паролем").size() == 1);
		CHECK(ocr.search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(ocr.search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(ocr.search("Снимок экрана").size() == 1);
		CHECK(ocr.search("Справка").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/desktop.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Веб-браузер").size() == 1);
		CHECK(ocr.search("Firefox").size() == 1);
		CHECK(ocr.search("Корзина").size() == 1);
		CHECK(ocr.search("Мой").size() == 1);
		CHECK(ocr.search("компьютер").size() == 1);
		CHECK(ocr.search("Помощь").size() == 1);
		CHECK(ocr.search("ASTRALINUX").size() == 1);
		CHECK(ocr.search("EN").size() == 1);
		CHECK(ocr.search("15:00").size() == 1);
		CHECK(ocr.search("Ср, 18 дек").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Операционная система").size() == 1);
		CHECK(ocr.search("общего назначения").size() == 1);
		CHECK(ocr.search("Орёл").size() == 1);
		CHECK(ocr.search("Разметка дисков").size() == 1);
		CHECK(ocr.search("Выберите диск для разметки").size() == 1);
		CHECK(ocr.search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2-alt.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Операционная система").size() == 1);
		CHECK(ocr.search("общего назначения").size() == 1);
		CHECK(ocr.search("Орёл").size() == 1);
		CHECK(ocr.search("Разметка дисков").size() == 1);
		CHECK(ocr.search("Выберите диск для разметки").size() == 1);
		CHECK(ocr.search("SCSI1 (0,0,0) (sda) - 10.7 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/grub.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(ocr.search("Внимание!").size() == 1);
		CHECK(ocr.search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(ocr.search("Нет").size() == 1);
		CHECK(ocr.search("Да").size() == 3);
		CHECK(ocr.search("Вернуться").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/incorrect_password.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Настройка учётных записей пользователей и паролей").size() == 1);
		CHECK(ocr.search("Неправильный пароль").size() == 1);
		CHECK(ocr.search("Введите пароль для нового администратора").size() == 1);
		CHECK(ocr.search("Введите пароль ещё раз").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/keyboard.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Настройка клавиатуры").size() == 1);
		CHECK(ocr.search("Способ переключения между национальной и латинской раскладкой").size() == 1);
		CHECK(ocr.search("Caps Lock").size() == 4);
		CHECK(ocr.search("правый Control").size() == 1);
		CHECK(ocr.search("Alt+Shift").size() == 3);
		CHECK(ocr.search("Control+Alt").size() == 1);
		CHECK(ocr.search("левая клавиша с логотипом").size() == 1);
		CHECK(ocr.search("без переключателя").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Вход в astra").size() == 1);
		CHECK(ocr.search("среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
		CHECK(ocr.search("user").size() == 1);
		CHECK(ocr.search("Имя:").size() == 1);
		CHECK(ocr.search("Пароль:").size() == 1);
		CHECK(ocr.search("Тип сессии").size() == 1);
		CHECK(ocr.search("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen_alt.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Вход в testo-astralinux").size() == 1);
		CHECK(ocr.search("четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
		CHECK(ocr.search("user").size() == 1);
		CHECK(ocr.search("Имя:").size() == 1);
		CHECK(ocr.search("Пароль:").size() == 1);
		CHECK(ocr.search("Тип сессии").size() == 1);
		CHECK(ocr.search("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/programs.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Выбор программного обеспечения").size() == 1);
		CHECK(ocr.search("Выберите устанавливаемое программное обеспечение").size() == 1);
		CHECK(ocr.search("Базовые средства").size() == 1);
		CHECK(ocr.search("Рабочий стол Fly").size() == 1);
		CHECK(ocr.search("Приложения для работы с сенсорным экраном").size() == 1);
		CHECK(ocr.search("СУБД").size() == 1);
		CHECK(ocr.search("Средства удаленного доступа SSH").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/time.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Настройка времени").size() == 1);
		CHECK(ocr.search("Выберите часовой пояс").size() == 1);
		CHECK(ocr.search("Москва").size() == 12);
		CHECK(ocr.search("Москва-01 - Калининград").size() == 1);
		CHECK(ocr.search("Москва+00 - Москва").size() == 1);
		CHECK(ocr.search("Москва+07 - Владивосток").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
	}
}

TEST_CASE("Astra Linux Orel/Установка (Консоль)") {
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/additional_options.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Дополнительные настройки ОС").size() == 2);
		CHECK(ocr.search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(ocr.search("Включить блокировку консоли").size() == 1);
		CHECK(ocr.search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(ocr.search("Использовать sudo с паролем").size() == 1);
		CHECK(ocr.search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(ocr.search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(ocr.search("Справка").size() == 1);
		CHECK(ocr.search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/disks-2.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Разметка дисков").size() == 1);
		CHECK(ocr.search("Выберите диск для разметки").size() == 1);
		CHECK(ocr.search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(ocr.search("Вернуться").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/grub.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(ocr.search("Внимание!").size() == 1);
		CHECK(ocr.search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(ocr.search("Нет").size() == 1);
		CHECK(ocr.search("Да").size() == 3);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/login.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("astra login:").size() == 2);
		CHECK(ocr.search("Password:").size() == 2);
		CHECK(ocr.search("Login incorrect").size() == 1);
		CHECK(ocr.search("You have mail.").size() == 1);
		CHECK(ocr.search("user@astra:~$").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/Start.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Графическая установка").size() == 1);
		CHECK(ocr.search("Установка").size() == 2);
		CHECK(ocr.search("Режим восстановления").size() == 1);
		CHECK(ocr.search("Орёл - город воинской славы").size() == 1);
		CHECK(ocr.search("Русский").size() == 1);
		CHECK(ocr.search("English").size() == 1);
		CHECK(ocr.search("F1 Язык").size() == 1);
		CHECK(ocr.search("F2 Параметры").size() == 1);
	}
}
