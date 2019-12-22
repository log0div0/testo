
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Континент 3/Установка") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Континент 3/Установка/Нажмите Enter чтобы установить вручную.png");
		CHECK(detector.detect(img, "Нажмите Enter чтобы провести инсталляцию вручную").size() == 1);
	}

	{
		stb::Image img("Континент 3/Установка/Отсутствует Соболь.png");
		CHECK(detector.detect(img, "bound to 192.168.122.145 -- renewal in 1720 seconds.").size() == 1);
		CHECK(detector.detect(img, "Отсутствует электронный замок Соболь").size() == 1);
		CHECK(detector.detect(img, "Дальнейшая работа будет производиться без него").size() == 1);
		CHECK(detector.detect(img, "продолжить? (y/n):").size() == 1);
	}

	{
		stb::Image img("Континент 3/Установка/Выберите вариант установки и действие.png");
		CHECK(detector.detect(img, "Выберите вариант установки:").size() == 1);
		CHECK(detector.detect(img, "4: ЦУС с сервером доступа (отладочная версия)").size() == 1);
		CHECK(detector.detect(img, "Введите номер варианта [1..7]:").size() == 1);
		CHECK(detector.detect(img, "Выберите действие").size() == 1);
		CHECK(detector.detect(img, "1: Установка").size() == 1);
		CHECK(detector.detect(img, "Введите номер варианта [1..3]:").size() == 1);
		CHECK(detector.detect(img, "Установка << Континент >>").size() == 1);
		CHECK(detector.detect(img, "продолжить? (y/n):").size() == 2);
	}

	{
		stb::Image img("Континент 3/Установка/Инициализация.png");
		CHECK(detector.detect(img, "Обнаруженные интерфейсы:").size() == 2);
		CHECK(detector.detect(img, "1.       em0").size() == 1);
		CHECK(detector.detect(img, "Укажите номер внешнего интерфейса:").size() == 1);
		CHECK(detector.detect(img, "Введите внешний IP адрес шлюза:").size() == 1);
		CHECK(detector.detect(img, "Продолжить? (Y/N):").size() == 3);
		CHECK(detector.detect(img, "2.       em1").size() == 2);
		CHECK(detector.detect(img, "Укажите номер внутреннего интерфейса.").size() == 1);
		CHECK(detector.detect(img, "Введите внутренний IP адрес шлюза:").size() == 1);
		CHECK(detector.detect(img, "Введите адрес маршрутизатора по умолчанию:").size() == 1);
		CHECK(detector.detect(img, "Использовать внешний носитель для инициализации? (Y/N):").size() == 1);
	}

	{
		stb::Image img("Континент 3/Установка/Главное меню после инициализации.png");
		CHECK(detector.detect(img, "Вставте носитель для записи ключа администратора ЦУС и нажмите Enter").size() == 1);
		CHECK(detector.detect(img, "Ключи администратора успешно сохранены").size() == 1);
		CHECK(detector.detect(img, "Создать учетную запись локального администратора? (Y/N):").size() == 1);
		CHECK(detector.detect(img, "Введите логин администратора:").size() == 1);
		CHECK(detector.detect(img, "Введите пароль:").size() == 1);
		CHECK(detector.detect(img, "Повторите пароль:").size() == 2);
		CHECK(detector.detect(img, "Конфигурация ЦУС завершена").size() == 1);
		CHECK(detector.detect(img, "1: Завершение работы").size() == 1);
		CHECK(detector.detect(img, "2: Перезагрузка").size() == 1);
		CHECK(detector.detect(img, "3: Управление конфигурацией").size() == 1);
		CHECK(detector.detect(img, "4: Настройка безопасности").size() == 1);
		CHECK(detector.detect(img, "5: Настройка СД").size() == 1);
		CHECK(detector.detect(img, "6: Тестирование").size() == 1);
		CHECK(detector.detect(img, "0: Выход").size() == 1);
		CHECK(detector.detect(img, "Выберите пункт меню (0 - 6):").size() == 1);
	}
}
