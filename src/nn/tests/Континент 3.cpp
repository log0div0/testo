
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Континент 3/Установка") {
	{
		stb::Image image("Континент 3/Установка/Нажмите Enter чтобы установить вручную.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Нажмите Enter чтобы провести инсталляцию вручную").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Отсутствует Соболь.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "bound to 192.168.122.145 -- renewal in 1720 seconds.").size() == 1);
		CHECK(tensor.match(&image, "Отсутствует электронный замок Соболь").size() == 1);
		CHECK(tensor.match(&image, "Дальнейшая работа будет производиться без него").size() == 1);
		CHECK(tensor.match(&image, "продолжить? (y/n):").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Выберите вариант установки и действие.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите вариант установки:").size() == 1);
		CHECK(tensor.match(&image, "4: ЦУС с сервером доступа (отладочная версия)").size() == 1);
		CHECK(tensor.match(&image, "Введите номер варианта [1..7]:").size() == 1);
		CHECK(tensor.match(&image, "Выберите действие").size() == 1);
		CHECK(tensor.match(&image, "1: Установка").size() == 1);
		CHECK(tensor.match(&image, "Введите номер варианта [1..3]:").size() == 1);
		CHECK(tensor.match(&image, "Установка << Континент >>").size() == 1);
		CHECK(tensor.match(&image, "продолжить? (y/n):").size() == 2);
	}

	{
		stb::Image image("Континент 3/Установка/Инициализация.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Обнаруженные интерфейсы:").size() == 2);
		CHECK(tensor.match(&image, "Укажите номер внешнего интерфейса:").size() == 1);
		CHECK(tensor.match(&image, "Введите внешний IP адрес шлюза:").size() == 1);
		CHECK(tensor.match(&image, "Продолжить? (Y/N):").size() == 3);
		CHECK(tensor.match(&image, "Укажите номер внутреннего интерфейса.").size() == 1);
		CHECK(tensor.match(&image, "Введите внутренний IP адрес шлюза:").size() == 1);
		CHECK(tensor.match(&image, "Введите адрес маршрутизатора по умолчанию:").size() == 1);
		CHECK(tensor.match(&image, "Использовать внешний носитель для инициализации? (Y/N):").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Главное меню после инициализации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Вставьте носитель для записи ключа администратора ЦУС и нажмите Enter").size() == 1);
		CHECK(tensor.match(&image, "Ключи администратора успешно сохранены").size() == 1);
		CHECK(tensor.match(&image, "Создать учетную запись локального администратора? (Y/N):").size() == 1);
		CHECK(tensor.match(&image, "Введите логин администратора:").size() == 1);
		CHECK(tensor.match(&image, "Введите пароль:").size() == 1);
		CHECK(tensor.match(&image, "Повторите пароль:").size() == 2);
		CHECK(tensor.match(&image, "Конфигурация ЦУС завершена").size() == 1);
		CHECK(tensor.match(&image, "1: Завершение работы").size() == 1);
		CHECK(tensor.match(&image, "2: Перезагрузка").size() == 1);
		CHECK(tensor.match(&image, "3: Управление конфигурацией").size() == 1);
		CHECK(tensor.match(&image, "4: Настройка безопасности").size() == 1);
		CHECK(tensor.match(&image, "5: Настройка СД").size() == 1);
		CHECK(tensor.match(&image, "6: Тестирование").size() == 1);
		CHECK(tensor.match(&image, "0: Выход").size() == 1);
		CHECK(tensor.match(&image, "Выберите пункт меню (0 - 6):").size() == 1);
	}
}
