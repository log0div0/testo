
#include <catch.hpp>
#include "nn/Context.hpp"

TEST_CASE("Astra Linux Orel/Установка (GUI)") {
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/additional_options.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Дополнительные настройки ОС").size() == 2);
		CHECK(context.ocr().search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(context.ocr().search("Включить блокировку консоли").size() == 1);
		CHECK(context.ocr().search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(context.ocr().search("Использовать sudo с паролем").size() == 1);
		CHECK(context.ocr().search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(context.ocr().search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(context.ocr().search("Снимок экрана").size() == 1);
		CHECK(context.ocr().search("Справка").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/desktop.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Веб-браузер").size() == 1);
		CHECK(context.ocr().search("Firefox").size() == 1);
		CHECK(context.ocr().search("Корзина").size() == 1);
		CHECK(context.ocr().search("Мой").size() == 1);
		CHECK(context.ocr().search("компьютер").size() == 1);
		CHECK(context.ocr().search("Помощь").size() == 1);
		CHECK(context.ocr().search("ASTRALINUX").size() == 1);
		CHECK(context.ocr().search("EN").size() == 1);
		CHECK(context.ocr().search("15:00").size() == 1);
		CHECK(context.ocr().search("Ср, 18 дек").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Операционная система").size() == 1);
		CHECK(context.ocr().search("общего назначения").size() == 1);
		CHECK(context.ocr().search("Орёл").size() == 1);
		CHECK(context.ocr().search("Разметка дисков").size() == 1);
		CHECK(context.ocr().search("Выберите диск для разметки").size() == 1);
		CHECK(context.ocr().search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2-alt.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Операционная система").size() == 1);
		CHECK(context.ocr().search("общего назначения").size() == 1);
		CHECK(context.ocr().search("Орёл").size() == 1);
		CHECK(context.ocr().search("Разметка дисков").size() == 1);
		CHECK(context.ocr().search("Выберите диск для разметки").size() == 1);
		CHECK(context.ocr().search("SCSI1 (0,0,0) (sda) - 10.7 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/grub.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(context.ocr().search("Внимание!").size() == 1);
		CHECK(context.ocr().search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(context.ocr().search("Нет").size() == 1);
		CHECK(context.ocr().search("Да").size() == 3);
		CHECK(context.ocr().search("Вернуться").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/incorrect_password.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Настройка учётных записей пользователей и паролей").size() == 1);
		CHECK(context.ocr().search("Неправильный пароль").size() == 1);
		CHECK(context.ocr().search("Введите пароль для нового администратора").size() == 1);
		CHECK(context.ocr().search("Введите пароль ещё раз").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/keyboard.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Настройка клавиатуры").size() == 1);
		CHECK(context.ocr().search("Способ переключения между национальной и латинской раскладкой").size() == 1);
		CHECK(context.ocr().search("Caps Lock").size() == 4);
		CHECK(context.ocr().search("правый Control").size() == 1);
		CHECK(context.ocr().search("Alt+Shift").size() == 3);
		CHECK(context.ocr().search("Control+Alt").size() == 1);
		CHECK(context.ocr().search("левая клавиша с логотипом").size() == 1);
		CHECK(context.ocr().search("без переключателя").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Вход в astra").size() == 1);
		CHECK(context.ocr().search("среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
		CHECK(context.ocr().search("user").size() == 1);
		CHECK(context.ocr().search("Имя:").size() == 1);
		CHECK(context.ocr().search("Пароль:").size() == 1);
		CHECK(context.ocr().search("Тип сессии").size() == 1);
		CHECK(context.ocr().search("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen_alt.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Вход в testo-astralinux").size() == 1);
		CHECK(context.ocr().search("четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
		CHECK(context.ocr().search("user").size() == 1);
		CHECK(context.ocr().search("Имя:").size() == 1);
		CHECK(context.ocr().search("Пароль:").size() == 1);
		CHECK(context.ocr().search("Тип сессии").size() == 1);
		CHECK(context.ocr().search("En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/programs.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выбор программного обеспечения").size() == 1);
		CHECK(context.ocr().search("Выберите устанавливаемое программное обеспечение").size() == 1);
		CHECK(context.ocr().search("Базовые средства").size() == 1);
		CHECK(context.ocr().search("Рабочий стол Fly").size() == 1);
		CHECK(context.ocr().search("Приложения для работы с сенсорным экраном").size() == 1);
		CHECK(context.ocr().search("СУБД").size() == 1);
		CHECK(context.ocr().search("Средства удаленного доступа SSH").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/time.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Настройка времени").size() == 1);
		CHECK(context.ocr().search("Выберите часовой пояс").size() == 1);
		CHECK(context.ocr().search("Москва").size() == 12);
		CHECK(context.ocr().search("Москва-01 - Калининград").size() == 1);
		CHECK(context.ocr().search("Москва+00 - Москва").size() == 1);
		CHECK(context.ocr().search("Москва+07 - Владивосток").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
	}
}

TEST_CASE("Astra Linux Orel/Установка (Консоль)") {
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/additional_options.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Дополнительные настройки ОС").size() == 2);
		CHECK(context.ocr().search("Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(context.ocr().search("Включить блокировку консоли").size() == 1);
		CHECK(context.ocr().search("Включить блокировку интерпретаторов").size() == 1);
		CHECK(context.ocr().search("Использовать sudo с паролем").size() == 1);
		CHECK(context.ocr().search("Отключить автоматическую настройку сети").size() == 2);
		CHECK(context.ocr().search("Установить 32-х битный загрузчик").size() == 1);
		CHECK(context.ocr().search("Справка").size() == 1);
		CHECK(context.ocr().search("Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/disks-2.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Разметка дисков").size() == 1);
		CHECK(context.ocr().search("Выберите диск для разметки").size() == 1);
		CHECK(context.ocr().search("SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(context.ocr().search("Вернуться").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/grub.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(context.ocr().search("Внимание!").size() == 1);
		CHECK(context.ocr().search("Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(context.ocr().search("Нет").size() == 1);
		CHECK(context.ocr().search("Да").size() == 3);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/login.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("astra login:").size() == 2);
		CHECK(context.ocr().search("Password:").size() == 2);
		CHECK(context.ocr().search("Login incorrect").size() == 1);
		CHECK(context.ocr().search("You have mail.").size() == 1);
		CHECK(context.ocr().search("user@astra:~$").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/Start.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Графическая установка").size() == 1);
		CHECK(context.ocr().search("Установка").size() == 2);
		CHECK(context.ocr().search("Режим восстановления").size() == 1);
		CHECK(context.ocr().search("Орёл - город воинской славы").size() == 1);
		CHECK(context.ocr().search("Русский").size() == 1);
		CHECK(context.ocr().search("English").size() == 1);
		CHECK(context.ocr().search("F1 Язык").size() == 1);
		CHECK(context.ocr().search("F2 Параметры").size() == 1);
	}
}
