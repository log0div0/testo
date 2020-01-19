
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 7/Установка") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Windows 7/Установка/Windows завершает применение параметров.png");
		CHECK(result.search("Настройка Windows").size() == 1);
		CHECK(result.search("Windows завершает применение параметров").size() == 1);
		CHECK(result.search("Справка").size() == 1);
		CHECK(result.search("Русский (Россия)").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Введите имя пользователя.png");
		CHECK(result.search("Выберите имя пользователя для вашей учетной записи, а также имя компьютера в сети.").size() == 1);
		CHECK(result.search("Введите имя пользователя (например, Андрей):").size() == 1);
		CHECK(result.search("Введите имя компьютера:").size() == 1);
		CHECK(result.search("Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Введите ключ продукта.png");
		CHECK(result.search("Введите ключ продукта Windows").size() == 1);
		CHECK(result.search("ключ продукта").size() == 3);
		CHECK(result.search("Наклейка с ключом продукта выглядит так:").size() == 1);
		CHECK(result.search("XXXXX-XXXXX-XXXXX-XXXXX-XXXXX").size() == 1);
		CHECK(result.search("Автоматически активировать Windows при подключении к Интернету").size() == 1);
		CHECK(result.search("Что такое активация?").size() == 1);
		CHECK(result.search("Заявление о конфиденциальности").size() == 1);
		CHECK(result.search("Пропустить").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Введите пароль.png");
		CHECK(result.search("Установите пароль для своей учетной записи").size() == 1);
		CHECK(result.search("Введите пароль (рекомендуется):").size() == 1);
		CHECK(result.search("Подтверждение пароля:").size() == 1);
		CHECK(result.search("Введите подсказку для пароля:").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выберите раздел для установки.png");
		CHECK(result.search("Выберите раздел для установки Windows").size() == 1);
		CHECK(result.search("Файл").size() == 1);
		CHECK(result.search("Полный раз...").size() == 1);
		CHECK(result.search("Свободно").size() == 1);
		CHECK(result.search("Тип").size() == 1);
		CHECK(result.search("Незанятое место на диске 0").size() == 1);
		CHECK(result.search("20.0 ГБ").size() == 2);
		CHECK(result.search("Обновить").size() == 1);
		CHECK(result.search("Настройка диска").size() == 1);
		CHECK(result.search("Загрузка").size() == 1);
		CHECK(result.search("Далее").size() == 1);
		CHECK(result.search("Сбор информации").size() == 1);
		CHECK(result.search("Установка Windows").size() == 2);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выберите тип установки.png");
		CHECK(result.search("Выберите тип установки").size() == 1);
		CHECK(result.search("Обновление").size() == 3);
		CHECK(result.search("Полная установка (дополнительные параметры)").size() == 1);
		CHECK(result.search("Помощь в принятии решения").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выбор версии.png");
		CHECK(result.search("Выберите операционную систему, которую следует установить").size() == 1);
		CHECK(result.search("Операционная система").size() == 1);
		CHECK(result.search("Архитектура").size() == 1);
		CHECK(result.search("Дата").size() == 1);
		CHECK(result.search("Windows 7 Начальная").size() == 2);
		CHECK(result.search("Windows 7 Домашняя базовая").size() == 2);
		CHECK(result.search("Windows 7 Домашняя расширенная").size() == 2);
		CHECK(result.search("Windows 7 Профессиональная").size() == 2);
		CHECK(result.search("Windows 7 Максимальная").size() == 2);
		CHECK(result.search("x86").size() == 5);
		CHECK(result.search("x64").size() == 4);
		CHECK(result.search("11/20/2010").size() == 5);
		CHECK(result.search("11/21/2010").size() == 4);
		CHECK(result.search("Описание:").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выбор языка.png");
		CHECK(result.search("My language is English").size() == 1);
		CHECK(result.search("Мой язык - русский").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выбор языка 2.png");
		CHECK(result.search("Устанавливаемый язык:").size() == 1);
		CHECK(result.search("Формат времени и денежных единиц:").size() == 1);
		CHECK(result.search("Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(result.search("Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Для продолжения требуется перезагрузка.png");
		CHECK(result.search("Для продолжения требуется перезагрузка Windows").size() == 1);
		CHECK(result.search("Перезагрузка через 5 сек.").size() == 1);
		CHECK(result.search("Перезагрузить сейчас").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Настройка даты и времени.png");
		CHECK(result.search("Проверьте настройку даты и времени").size() == 1);
		CHECK(result.search("Часовой пояс:").size() == 1);
		CHECK(result.search("Автоматический переход на летнее время и обратно").size() == 1);
		CHECK(result.search("Дата:").size() == 1);
		CHECK(result.search("Время:").size() == 1);
		CHECK(result.search("Ноябрь 2019").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Настройка сети.png");
		CHECK(result.search("Выберите текущее место расположения компьютера").size() == 1);
		CHECK(result.search("Домашняя сеть").size() == 1);
		CHECK(result.search("Рабочая сеть").size() == 1);
		CHECK(result.search("Общественная сеть").size() == 1);
		CHECK(result.search("Если не уверены, выбирайте общественную сеть.").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Ознакомьтесь с лицензией.png");
		CHECK(result.search("Ознакомьтесь с условиями лицензии").size() == 1);
		CHECK(result.search("Условия лицензии на программное обеспечение Microsoft").size() == 1);
		CHECK(result.search("Windows 7 Максимальная с пакетом обновления 1").size() == 1);
		CHECK(result.search("Я принимаю условия лицензии").size() == 1);
		CHECK(result.search("Далее").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Помогите автоматически защитить компьютер.png");
		CHECK(result.search("Помогите автоматически защитить компьютер и улучшить Windows").size() == 1);
		CHECK(result.search("Использовать рекомендуемые параметры").size() == 1);
		CHECK(result.search("Устанавливать только наиболее важные обновления").size() == 1);
		CHECK(result.search("Отложить решение").size() == 1);
		CHECK(result.search("Подробнее об этих параметрах").size() == 1);
		CHECK(result.search("Заявление о конфиденциальности").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Просто кнопка Установить.png");
		CHECK(result.search("Установка Windows").size() == 1);
		CHECK(result.search("Windows 7").size() == 1);
		CHECK(result.search("Установить").size() == 1);
		CHECK(result.search("Что следует знать перед выполнением установки Windows").size() == 1);
		CHECK(result.search("Восстановление системы").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Рабочий стол сразу после установки.png");
		CHECK(result.search("Корзина").size() == 1);
		CHECK(result.search("RU").size() == 1);
		CHECK(result.search("13:56").size() == 1);
		CHECK(result.search("23.11.2019").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Ход установки.png");
		CHECK(result.search("Установка Windows...").size() == 1);
		CHECK(result.search("Копирование файлов Windows").size() == 1);
		CHECK(result.search("Распаковка файлов Windows (0%)").size() == 1);
		CHECK(result.search("Установка компонентов").size() == 1);
		CHECK(result.search("Установка обновлений").size() == 1);
		CHECK(result.search("Завершение установки").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Экран входа в систему.png");
		CHECK(result.search("RU").size() == 1);
		CHECK(result.search("Петя").size() == 1);
		CHECK(result.search("Пароль").size() == 1);
		CHECK(result.search("Windows 7 Максимальная").size() == 1);
	}
}
