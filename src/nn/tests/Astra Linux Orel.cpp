
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Astra Linux Orel/Установка (GUI)") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/additional_options.png");
		CHECK(result.search("Дополнительные настройки ОС").size() == 2);
		CHECK(result.search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(result.search("Включить блокировку консоли").size() == 1);
		CHECK(result.search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(result.search("Использовать sudo с паролем").size() == 1);
		CHECK(result.search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(result.search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(result.search("Снимок экрана").size() == 1);
		CHECK(result.search("Справка").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/desktop.png");
		CHECK(result.search("Веб-браузер").size() == 1);
		CHECK(result.search("Firefox").size() == 1);
		CHECK(result.search("Корзина").size() == 1);
		CHECK(result.search("Мой").size() == 1);
		CHECK(result.search("компьютер").size() == 1);
		CHECK(result.search("Помощь").size() == 1);
		CHECK(result.search("ASTRALINUX").size() == 1);
		CHECK(result.search("EN").size() == 1);
		CHECK(result.search("15:00").size() == 1);
		CHECK(result.search("Ср, 18 дек").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/disks-2.png");
		CHECK(result.search("Операционная система").size() == 1);
		CHECK(result.search("общего назначения").size() == 1);
		CHECK(result.search("Орёл").size() == 1);
		CHECK(result.search("Разметка дисков").size() == 1);
		CHECK(result.search("Выберите диск для разметки").size() == 1);
		CHECK(result.search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/disks-2-alt.png");
		CHECK(result.search("Операционная система").size() == 1);
		CHECK(result.search("общего назначения").size() == 1);
		CHECK(result.search("Орёл").size() == 1);
		CHECK(result.search("Разметка дисков").size() == 1);
		CHECK(result.search("Выберите диск для разметки").size() == 1);
		CHECK(result.search("SCSI1 (0,0,0) (sda) - 10.7 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/grub.png");
		CHECK(result.search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(result.search("Внимание!").size() == 1);
		CHECK(result.search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(result.search("Нет").size() == 1);
		CHECK(result.search("Да").size() == 3);
		CHECK(result.search("Вернуться").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/incorrect_password.png");
		CHECK(result.search("Настройка учётных записей пользователей и паролей").size() == 1);
		CHECK(result.search("Неправильный пароль").size() == 1);
		CHECK(result.search("Введите пароль для нового администратора").size() == 1);
		CHECK(result.search("Введите пароль ещё раз").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/keyboard.png");
		CHECK(result.search("Настройка клавиатуры").size() == 1);
		CHECK(result.search("Способ переключения между национальной и латинской раскладкой").size() == 1);
		CHECK(result.search("Caps Lock").size() == 4);
		CHECK(result.search("правый Control").size() == 1);
		CHECK(result.search("Alt+Shift").size() == 3);
		CHECK(result.search("Control+Alt").size() == 1);
		CHECK(result.search("левая клавиша с логотипом").size() == 1);
		CHECK(result.search("без переключателя").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/login_screen.png");
		CHECK(result.search("Вход в astra").size() == 1);
		CHECK(result.search("среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
		CHECK(result.search("user").size() == 1);
		CHECK(result.search("Имя:").size() == 1);
		CHECK(result.search("Пароль:").size() == 1);
		CHECK(result.search("Тип сессии").size() == 1);
		CHECK(result.search("En").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/login_screen_alt.png");
		CHECK(result.search("Вход в testo-astralinux").size() == 1);
		CHECK(result.search("четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
		CHECK(result.search("user").size() == 1);
		CHECK(result.search("Имя:").size() == 1);
		CHECK(result.search("Пароль:").size() == 1);
		CHECK(result.search("Тип сессии").size() == 1);
		CHECK(result.search("En").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/programs.png");
		CHECK(result.search("Выбор программного обеспечения").size() == 1);
		CHECK(result.search("Выберите устанавливаемое программное обеспечение").size() == 1);
		CHECK(result.search("Базовые средства").size() == 1);
		CHECK(result.search("Рабочий стол Fly").size() == 1);
		CHECK(result.search("Приложения для работы с сенсорным экраном").size() == 1);
		CHECK(result.search("СУБД").size() == 1);
		CHECK(result.search("Средства удаленного доступа SSH").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (GUI)/time.png");
		CHECK(result.search("Настройка времени").size() == 1);
		CHECK(result.search("Выберите часовой пояс").size() == 1);
		CHECK(result.search("Москва").size() == 12);
		CHECK(result.search("Москва-01 - Калининград").size() == 1);
		CHECK(result.search("Москва+00 - Москва").size() == 1);
		CHECK(result.search("Москва+07 - Владивосток").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
	}
}

TEST_CASE("Astra Linux Orel/Установка (Консоль)") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Astra Linux Orel/Установка (Консоль)/additional_options.png");
		CHECK(result.search("Дополнительные настройки ОС").size() == 2);
		CHECK(result.search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(result.search("Включить блокировку консоли").size() == 1);
		CHECK(result.search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(result.search("Использовать sudo с паролем").size() == 1);
		CHECK(result.search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(result.search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(result.search("Справка").size() == 1);
		CHECK(result.search("Продолжить").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (Консоль)/disks-2.png");
		CHECK(result.search("Разметка дисков").size() == 1);
		CHECK(result.search("Выберите диск для разметки").size() == 1);
		CHECK(result.search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(result.search("Вернуться").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (Консоль)/grub.png");
		CHECK(result.search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(result.search("Внимание!").size() == 1);
		CHECK(result.search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(result.search("Нет").size() == 1);
		CHECK(result.search("Да").size() == 3);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (Консоль)/login.png");
		CHECK(result.search("astra login:").size() == 2);
		CHECK(result.search("Password:").size() == 2);
		CHECK(result.search("Login incorrect").size() == 1);
		CHECK(result.search("You have mail.").size() == 1);
		CHECK(result.search("user@astra:~$").size() == 1);
	}
	{
		auto result = ocr.run("Astra Linux Orel/Установка (Консоль)/Start.png");
		CHECK(result.search("Графическая установка").size() == 1);
		CHECK(result.search("Установка").size() == 2);
		CHECK(result.search("Режим восстановления").size() == 1);
		CHECK(result.search("Орёл - город воинской славы").size() == 1);
		CHECK(result.search("Русский").size() == 1);
		CHECK(result.search("English").size() == 1);
		CHECK(result.search("F1 Язык").size() == 1);
		CHECK(result.search("F2 Параметры").size() == 1);
	}
}
