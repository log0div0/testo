
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 7/Установка") {
	{
		stb::Image image("Windows 7/Установка/Windows завершает применение параметров.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Настройка Windows").size() == 1);
		CHECK(tensor.match("Windows завершает применение параметров").size() == 1);
		CHECK(tensor.match("Справка").size() == 1);
		CHECK(tensor.match("Русский (Россия)").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите имя пользователя.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Выберите имя пользователя для вашей учетной записи, а также имя компьютера в сети.").size() == 1);
		CHECK(tensor.match("Введите имя пользователя (например, Андрей):").size() == 1);
		CHECK(tensor.match("Введите имя компьютера:").size() == 1);
		CHECK(tensor.match("Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.").size() == 1);
		CHECK(tensor.match("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите ключ продукта.png");
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
	{
		stb::Image image("Windows 7/Установка/Введите пароль.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установите пароль для своей учетной записи").size() == 1);
		CHECK(tensor.match("Введите пароль (рекомендуется):").size() == 1);
		CHECK(tensor.match("Подтверждение пароля:").size() == 1);
		CHECK(tensor.match("Введите подсказку для пароля:").size() == 1);
		CHECK(tensor.match("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выберите раздел для установки.png");
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
	{
		stb::Image image("Windows 7/Установка/Выберите тип установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Выберите тип установки").size() == 1);
		CHECK(tensor.match("Обновление").size() == 3);
		CHECK(tensor.match("Полная установка (дополнительные параметры)").size() == 1);
		CHECK(tensor.match("Помощь в принятии решения").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор версии.png");
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
	{
		stb::Image image("Windows 7/Установка/Выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("My language is English").size() == 1);
		CHECK(tensor.match("Мой язык - русский").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор языка 2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Устанавливаемый язык:").size() == 1);
		CHECK(tensor.match("Формат времени и денежных единиц:").size() == 1);
		CHECK(tensor.match("Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(tensor.match("Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Для продолжения требуется перезагрузка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Для продолжения требуется перезагрузка Windows").size() == 1);
		CHECK(tensor.match("Перезагрузка через 5 сек.").size() == 1);
		CHECK(tensor.match("Перезагрузить сейчас").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка даты и времени.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Проверьте настройку даты и времени").size() == 1);
		CHECK(tensor.match("Часовой пояс:").size() == 1);
		CHECK(tensor.match("Автоматический переход на летнее время и обратно").size() == 1);
		CHECK(tensor.match("Дата:").size() == 1);
		CHECK(tensor.match("Время:").size() == 1);
		CHECK(tensor.match("Ноябрь 2019").size() == 1);
		CHECK(tensor.match("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка сети.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Выберите текущее место расположения компьютера").size() == 1);
		CHECK(tensor.match("Домашняя сеть").size() == 1);
		CHECK(tensor.match("Рабочая сеть").size() == 1);
		CHECK(tensor.match("Общественная сеть").size() == 1);
		CHECK(tensor.match("Если не уверены, выбирайте общественную сеть.").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ознакомьтесь с лицензией.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Ознакомьтесь с условиями лицензии").size() == 1);
		CHECK(tensor.match("УСЛОВИЯ ЛИЦЕНЗИИ НА ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ MICROSOFT").size() == 1);
		CHECK(tensor.match("WINDOWS 7 МАКСИМАЛЬНАЯ С ПАКЕТОМ ОБНОВЛЕНИЯ 1").size() == 1);
		CHECK(tensor.match("Я принимаю условия лицензии").size() == 1);
		CHECK(tensor.match("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Помогите автоматически защитить компьютер.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Помогите автоматически защитить компьютер и улучшить Windows").size() == 1);
		CHECK(tensor.match("Использовать рекомендуемые параметры").size() == 1);
		CHECK(tensor.match("Устанавливать только наиболее важные обновления").size() == 1);
		CHECK(tensor.match("Отложить решение").size() == 1);
		CHECK(tensor.match("Подробнее об этих параметрах").size() == 1);
		CHECK(tensor.match("Заявление о конфиденциальности").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Просто кнопка Установить.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установка Windows").size() == 1);
		CHECK(tensor.match("Windows 7").size() == 1);
		CHECK(tensor.match("Установить").size() == 1);
		CHECK(tensor.match("Что следует знать перед выполнением установки Windows").size() == 1);
		CHECK(tensor.match("Восстановление системы").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Рабочий стол сразу после установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Корзина").size() == 1);
		CHECK(tensor.match("RU").size() == 1);
		CHECK(tensor.match("13:56").size() == 1);
		CHECK(tensor.match("23.11.2019").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ход установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установка Windows...").size() == 1);
		CHECK(tensor.match("Копирование файлов Windows").size() == 1);
		CHECK(tensor.match("Распаковка файлов Windows (0%)").size() == 1);
		CHECK(tensor.match("Установка компонентов").size() == 1);
		CHECK(tensor.match("Установка обновлений").size() == 1);
		CHECK(tensor.match("Завершение установки").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Экран входа в систему.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("RU").size() == 1);
		CHECK(tensor.match("Петя").size() == 1);
		CHECK(tensor.match("Пароль").size() == 1);
		CHECK(tensor.match("Windows 7 Максимальная").size() == 1);
	}
}

TEST_CASE("Windows 7/Континент CSP") {
	{
		stb::Image image("Windows 7/Континент CSP/Набор энтропии.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Для создание шаров-мишеней").size() == 1);
		CHECK(tensor.match("нажимайте левой кнопкой мыши").size() == 1);
		CHECK(tensor.match("пока индикатор не заполнится").size() == 1);
		CHECK(tensor.match("целиком").size() == 1);
		CHECK(tensor.match("Корзина").size() == 1);
		CHECK(tensor.match("Континент").size() == 1);
		CHECK(tensor.match("TLS-клиент").size() == 1);
	}
}
