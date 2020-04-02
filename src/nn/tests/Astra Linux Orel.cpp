
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Astra Linux Orel/Установка (GUI)") {
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/additional_options.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Дополнительные настройки ОС").size() == 2);
		CHECK(tensor.match(&image, "Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(tensor.match(&image, "Включить блокировку консоли").size() == 1);
		CHECK(tensor.match(&image, "Включить блокировку интерпретаторов").size() == 1);
		CHECK(tensor.match(&image, "Использовать sudo с паролем").size() == 1);
		CHECK(tensor.match(&image, "Отключить автоматическую настройку сети").size() == 2);
		CHECK(tensor.match(&image, "Установить 32-х битный загрузчик").size() == 1);
		CHECK(tensor.match(&image, "Снимок экрана").size() == 1);
		CHECK(tensor.match(&image, "Справка").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/desktop.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Веб-браузер").size() == 1);
		CHECK(tensor.match(&image, "Firefox").size() == 1);
		CHECK(tensor.match(&image, "Корзина").size() == 1);
		CHECK(tensor.match(&image, "Мой").size() == 1);
		CHECK(tensor.match(&image, "компьютер").size() == 1);
		CHECK(tensor.match(&image, "Помощь").size() == 1);
		CHECK(tensor.match(&image, "ASTRALINUX").size() == 1);
		CHECK(tensor.match(&image, "EN").size() == 1);
		CHECK(tensor.match(&image, "15:00").size() == 1);
		CHECK(tensor.match(&image, "Ср, 18 дек").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Операционная система").size() == 1);
		CHECK(tensor.match(&image, "общего назначения").size() == 1);
		CHECK(tensor.match(&image, "Орёл").size() == 1);
		CHECK(tensor.match(&image, "Разметка дисков").size() == 1);
		CHECK(tensor.match(&image, "Выберите диск для разметки").size() == 1);
		CHECK(tensor.match(&image, "SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/disks-2-alt.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Операционная система").size() == 1);
		CHECK(tensor.match(&image, "общего назначения").size() == 1);
		CHECK(tensor.match(&image, "Орёл").size() == 1);
		CHECK(tensor.match(&image, "Разметка дисков").size() == 1);
		CHECK(tensor.match(&image, "Выберите диск для разметки").size() == 1);
		CHECK(tensor.match(&image, "SCSI1 (0,0,0) (sda) - 10.7 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/grub.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(tensor.match(&image, "Внимание!").size() == 1);
		CHECK(tensor.match(&image, "Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(tensor.match(&image, "Нет").size() == 1);
		CHECK(tensor.match(&image, "Да").size() == 3);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/incorrect_password.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Настройка учётных записей пользователей и паролей").size() == 1);
		CHECK(tensor.match(&image, "Неправильный пароль").size() == 1);
		CHECK(tensor.match(&image, "Введите пароль для нового администратора").size() == 1);
		CHECK(tensor.match(&image, "Введите пароль ещё раз").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/keyboard.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Настройка клавиатуры").size() == 1);
		CHECK(tensor.match(&image, "Способ переключения между национальной и латинской раскладкой").size() == 1);
		CHECK(tensor.match(&image, "Caps Lock").size() == 4);
		CHECK(tensor.match(&image, "правый Control").size() == 1);
		CHECK(tensor.match(&image, "Alt+Shift").size() == 3);
		CHECK(tensor.match(&image, "Control+Alt").size() == 1);
		CHECK(tensor.match(&image, "левая клавиша с логотипом").size() == 1);
		CHECK(tensor.match(&image, "без переключателя").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Вход в astra").size() == 1);
		CHECK(tensor.match(&image, "среда, 18 декабря 2019 г. 14:59:36 MSK").size() == 1);
		CHECK(tensor.match(&image, "user").size() == 1);
		CHECK(tensor.match(&image, "Имя:").size() == 1);
		CHECK(tensor.match(&image, "Пароль:").size() == 1);
		CHECK(tensor.match(&image, "Тип сессии").size() == 1);
		CHECK(tensor.match(&image, "En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/login_screen_alt.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Вход в testo-astralinux").size() == 1);
		CHECK(tensor.match(&image, "четверг, 6 февраля 2020 г. 16:39:23 MSK").size() == 1);
		CHECK(tensor.match(&image, "user").size() == 1);
		CHECK(tensor.match(&image, "Имя:").size() == 1);
		CHECK(tensor.match(&image, "Пароль:").size() == 1);
		CHECK(tensor.match(&image, "Тип сессии").size() == 1);
		CHECK(tensor.match(&image, "En").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/programs.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выбор программного обеспечения").size() == 1);
		CHECK(tensor.match(&image, "Выберите устанавливаемое программное обеспечение").size() == 1);
		CHECK(tensor.match(&image, "Базовые средства").size() == 1);
		CHECK(tensor.match(&image, "Рабочий стол Fly").size() == 1);
		CHECK(tensor.match(&image, "Приложения для работы с сенсорным экраном").size() == 1);
		CHECK(tensor.match(&image, "СУБД").size() == 1);
		CHECK(tensor.match(&image, "Средства удаленного доступа SSH").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (GUI)/time.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Настройка времени").size() == 1);
		CHECK(tensor.match(&image, "Выберите часовой пояс").size() == 1);
		CHECK(tensor.match(&image, "Москва").size() == 12);
		CHECK(tensor.match(&image, "Москва-01 - Калининград").size() == 1);
		CHECK(tensor.match(&image, "Москва+00 - Москва").size() == 1);
		CHECK(tensor.match(&image, "Москва+07 - Владивосток").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
	}
}

TEST_CASE("Astra Linux Orel/Установка (Консоль)") {
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/additional_options.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Дополнительные настройки ОС").size() == 2);
		CHECK(tensor.match(&image, "Использовать по умолчанию ядро Hardened").size() == 1);
		CHECK(tensor.match(&image, "Включить блокировку консоли").size() == 1);
		CHECK(tensor.match(&image, "Включить блокировку интерпретаторов").size() == 1);
		CHECK(tensor.match(&image, "Использовать sudo с паролем").size() == 1);
		CHECK(tensor.match(&image, "Отключить автоматическую настройку сети").size() == 2);
		CHECK(tensor.match(&image, "Установить 32-х битный загрузчик").size() == 1);
		CHECK(tensor.match(&image, "Справка").size() == 1);
		CHECK(tensor.match(&image, "Продолжить").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/disks-2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Разметка дисков").size() == 1);
		CHECK(tensor.match(&image, "Выберите диск для разметки").size() == 1);
		CHECK(tensor.match(&image, "SCSI1 (0,0,0) (sda) - 21.5 GB ATA QEMU HARDDISK").size() == 1);
		CHECK(tensor.match(&image, "Вернуться").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/grub.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установка системного загрузчика GRUB на жёсткий диск").size() == 1);
		CHECK(tensor.match(&image, "Внимание!").size() == 1);
		CHECK(tensor.match(&image, "Установить системный загрузчик GRUB в главную загрузочную запись?").size() == 1);
		CHECK(tensor.match(&image, "Нет").size() == 1);
		CHECK(tensor.match(&image, "Да").size() == 3);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/login.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "astra login:").size() == 2);
		CHECK(tensor.match(&image, "Password:").size() == 2);
		CHECK(tensor.match(&image, "Login incorrect").size() == 1);
		CHECK(tensor.match(&image, "You have mail.").size() == 1);
		CHECK(tensor.match(&image, "user@astra:~$").size() == 1);
	}
	{
		stb::Image image("Astra Linux Orel/Установка (Консоль)/Start.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Графическая установка").size() == 1);
		CHECK(tensor.match(&image, "Установка").size() == 2);
		CHECK(tensor.match(&image, "Режим восстановления").size() == 1);
		CHECK(tensor.match(&image, "Орёл - город воинской славы").size() == 1);
		CHECK(tensor.match(&image, "Русский").size() == 1);
		CHECK(tensor.match(&image, "English").size() == 1);
		CHECK(tensor.match(&image, "F1 Язык").size() == 1);
		CHECK(tensor.match(&image, "F2 Параметры").size() == 1);
	}
}
