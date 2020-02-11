
#include <catch.hpp>
#include "nn/Context.hpp"

TEST_CASE("Windows 7/Установка") {
	{
		stb::Image image("Windows 7/Установка/Windows завершает применение параметров.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Настройка Windows").size() == 1);
		CHECK(context.ocr().search("Windows завершает применение параметров").size() == 1);
		CHECK(context.ocr().search("Справка").size() == 1);
		CHECK(context.ocr().search("Русский (Россия)").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите имя пользователя.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите имя пользователя для вашей учетной записи, а также имя компьютера в сети.").size() == 1);
		CHECK(context.ocr().search("Введите имя пользователя (например, Андрей):").size() == 1);
		CHECK(context.ocr().search("Введите имя компьютера:").size() == 1);
		CHECK(context.ocr().search("Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите ключ продукта.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Введите ключ продукта Windows").size() == 1);
		CHECK(context.ocr().search("ключ продукта").size() == 2);
		CHECK(context.ocr().search("КЛЮЧ ПРОДУКТА").size() == 1);
		CHECK(context.ocr().search("Наклейка с ключом продукта выглядит так:").size() == 1);
		CHECK(context.ocr().search("XXXXX-XXXXX-XXXXX-XXXXX-XXXXX").size() == 1);
		CHECK(context.ocr().search("Автоматически активировать Windows при подключении к Интернету").size() == 1);
		CHECK(context.ocr().search("Что такое активация?").size() == 1);
		CHECK(context.ocr().search("Заявление о конфиденциальности").size() == 1);
		CHECK(context.ocr().search("Пропустить").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите пароль.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Установите пароль для своей учетной записи").size() == 1);
		CHECK(context.ocr().search("Введите пароль (рекомендуется):").size() == 1);
		CHECK(context.ocr().search("Подтверждение пароля:").size() == 1);
		CHECK(context.ocr().search("Введите подсказку для пароля:").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выберите раздел для установки.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите раздел для установки Windows").size() == 1);
		CHECK(context.ocr().search("Файл").size() == 1);
		CHECK(context.ocr().search("Полный раз...").size() == 1);
		CHECK(context.ocr().search("Свободно").size() == 1);
		CHECK(context.ocr().search("Тип").size() == 1);
		CHECK(context.ocr().search("Незанятое место на диске 0").size() == 1);
		CHECK(context.ocr().search("20.0 ГБ").size() == 2);
		CHECK(context.ocr().search("Обновить").size() == 1);
		CHECK(context.ocr().search("Настройка диска").size() == 1);
		CHECK(context.ocr().search("Загрузка").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
		CHECK(context.ocr().search("Сбор информации").size() == 1);
		CHECK(context.ocr().search("Установка Windows").size() == 2);
	}
	{
		stb::Image image("Windows 7/Установка/Выберите тип установки.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите тип установки").size() == 1);
		CHECK(context.ocr().search("Обновление").size() == 3);
		CHECK(context.ocr().search("Полная установка (дополнительные параметры)").size() == 1);
		CHECK(context.ocr().search("Помощь в принятии решения").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор версии.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите операционную систему, которую следует установить").size() == 1);
		CHECK(context.ocr().search("Операционная система").size() == 1);
		CHECK(context.ocr().search("Архитектура").size() == 1);
		CHECK(context.ocr().search("Дата").size() == 1);
		CHECK(context.ocr().search("Windows 7 Начальная").size() == 2);
		CHECK(context.ocr().search("Windows 7 Домашняя базовая").size() == 2);
		CHECK(context.ocr().search("Windows 7 Домашняя расширенная").size() == 2);
		CHECK(context.ocr().search("Windows 7 Профессиональная").size() == 2);
		CHECK(context.ocr().search("Windows 7 Максимальная").size() == 2);
		CHECK(context.ocr().search("x86").size() == 5);
		CHECK(context.ocr().search("x64").size() == 4);
		CHECK(context.ocr().search("11/20/2010").size() == 5);
		CHECK(context.ocr().search("11/21/2010").size() == 4);
		CHECK(context.ocr().search("Описание:").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор языка.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("My language is English").size() == 1);
		CHECK(context.ocr().search("Мой язык - русский").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор языка 2.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Устанавливаемый язык:").size() == 1);
		CHECK(context.ocr().search("Формат времени и денежных единиц:").size() == 1);
		CHECK(context.ocr().search("Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(context.ocr().search("Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Для продолжения требуется перезагрузка.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Для продолжения требуется перезагрузка Windows").size() == 1);
		CHECK(context.ocr().search("Перезагрузка через 5 сек.").size() == 1);
		CHECK(context.ocr().search("Перезагрузить сейчас").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка даты и времени.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Проверьте настройку даты и времени").size() == 1);
		CHECK(context.ocr().search("Часовой пояс:").size() == 1);
		CHECK(context.ocr().search("Автоматический переход на летнее время и обратно").size() == 1);
		CHECK(context.ocr().search("Дата:").size() == 1);
		CHECK(context.ocr().search("Время:").size() == 1);
		CHECK(context.ocr().search("Ноябрь 2019").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка сети.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Выберите текущее место расположения компьютера").size() == 1);
		CHECK(context.ocr().search("Домашняя сеть").size() == 1);
		CHECK(context.ocr().search("Рабочая сеть").size() == 1);
		CHECK(context.ocr().search("Общественная сеть").size() == 1);
		CHECK(context.ocr().search("Если не уверены, выбирайте общественную сеть.").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ознакомьтесь с лицензией.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Ознакомьтесь с условиями лицензии").size() == 1);
		CHECK(context.ocr().search("УСЛОВИЯ ЛИЦЕНЗИИ НА ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ MICROSOFT").size() == 1);
		CHECK(context.ocr().search("WINDOWS 7 МАКСИМАЛЬНАЯ С ПАКЕТОМ ОБНОВЛЕНИЯ 1").size() == 1);
		CHECK(context.ocr().search("Я принимаю условия лицензии").size() == 1);
		CHECK(context.ocr().search("Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Помогите автоматически защитить компьютер.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Помогите автоматически защитить компьютер и улучшить Windows").size() == 1);
		CHECK(context.ocr().search("Использовать рекомендуемые параметры").size() == 1);
		CHECK(context.ocr().search("Устанавливать только наиболее важные обновления").size() == 1);
		CHECK(context.ocr().search("Отложить решение").size() == 1);
		CHECK(context.ocr().search("Подробнее об этих параметрах").size() == 1);
		CHECK(context.ocr().search("Заявление о конфиденциальности").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Просто кнопка Установить.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Установка Windows").size() == 1);
		CHECK(context.ocr().search("Windows 7").size() == 1);
		CHECK(context.ocr().search("Установить").size() == 1);
		CHECK(context.ocr().search("Что следует знать перед выполнением установки Windows").size() == 1);
		CHECK(context.ocr().search("Восстановление системы").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Рабочий стол сразу после установки.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Корзина").size() == 1);
		CHECK(context.ocr().search("RU").size() == 1);
		CHECK(context.ocr().search("13:56").size() == 1);
		CHECK(context.ocr().search("23.11.2019").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ход установки.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Установка Windows...").size() == 1);
		CHECK(context.ocr().search("Копирование файлов Windows").size() == 1);
		CHECK(context.ocr().search("Распаковка файлов Windows (0%)").size() == 1);
		CHECK(context.ocr().search("Установка компонентов").size() == 1);
		CHECK(context.ocr().search("Установка обновлений").size() == 1);
		CHECK(context.ocr().search("Завершение установки").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Экран входа в систему.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("RU").size() == 1);
		CHECK(context.ocr().search("Петя").size() == 1);
		CHECK(context.ocr().search("Пароль").size() == 1);
		CHECK(context.ocr().search("Windows 7 Максимальная").size() == 1);
	}
}
