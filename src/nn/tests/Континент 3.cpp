
#include <catch.hpp>
#include "nn/Context.hpp"

TEST_CASE("Континент 3/Установка") {
	{
		stb::Image image("Континент 3/Установка/Нажмите Enter чтобы установить вручную.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Нажмите Enter чтобы провести инсталляцию вручную").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Отсутствует Соболь.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("bound to 192.168.122.145 -- renewal in 1720 seconds.").size() == 1);
		CHECK(context.ocr().search("Отсутствует электронный замок Соболь").size() == 1);
		CHECK(context.ocr().search("Дальнейшая работа будет производиться без него").size() == 1);
		CHECK(context.ocr().search("продолжить? (y/n):").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Выберите вариант установки и действие.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите вариант установки:").size() == 1);
		CHECK(context.ocr().search("4: ЦУС с сервером доступа (отладочная версия)").size() == 1);
		CHECK(context.ocr().search("Введите номер варианта [1..7]:").size() == 1);
		CHECK(context.ocr().search("Выберите действие").size() == 1);
		CHECK(context.ocr().search("1: Установка").size() == 1);
		CHECK(context.ocr().search("Введите номер варианта [1..3]:").size() == 1);
		CHECK(context.ocr().search("Установка << Континент >>").size() == 1);
		CHECK(context.ocr().search("продолжить? (y/n):").size() == 2);
	}

	{
		stb::Image image("Континент 3/Установка/Инициализация.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Обнаруженные интерфейсы:").size() == 2);
		CHECK(context.ocr().search("Укажите номер внешнего интерфейса:").size() == 1);
		CHECK(context.ocr().search("Введите внешний IP адрес шлюза:").size() == 1);
		CHECK(context.ocr().search("Продолжить? (Y/N):").size() == 3);
		CHECK(context.ocr().search("Укажите номер внутреннего интерфейса.").size() == 1);
		CHECK(context.ocr().search("Введите внутренний IP адрес шлюза:").size() == 1);
		CHECK(context.ocr().search("Введите адрес маршрутизатора по умолчанию:").size() == 1);
		CHECK(context.ocr().search("Использовать внешний носитель для инициализации? (Y/N):").size() == 1);
	}

	{
		stb::Image image("Континент 3/Установка/Главное меню после инициализации.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Вставьте носитель для записи ключа администратора ЦУС и нажмите Enter").size() == 1);
		CHECK(context.ocr().search("Ключи администратора успешно сохранены").size() == 1);
		CHECK(context.ocr().search("Создать учетную запись локального администратора? (Y/N):").size() == 1);
		CHECK(context.ocr().search("Введите логин администратора:").size() == 1);
		CHECK(context.ocr().search("Введите пароль:").size() == 1);
		CHECK(context.ocr().search("Повторите пароль:").size() == 2);
		CHECK(context.ocr().search("Конфигурация ЦУС завершена").size() == 1);
		CHECK(context.ocr().search("1: Завершение работы").size() == 1);
		CHECK(context.ocr().search("2: Перезагрузка").size() == 1);
		CHECK(context.ocr().search("3: Управление конфигурацией").size() == 1);
		CHECK(context.ocr().search("4: Настройка безопасности").size() == 1);
		CHECK(context.ocr().search("5: Настройка СД").size() == 1);
		CHECK(context.ocr().search("6: Тестирование").size() == 1);
		CHECK(context.ocr().search("0: Выход").size() == 1);
		CHECK(context.ocr().search("Выберите пункт меню (0 - 6):").size() == 1);
	}
}
