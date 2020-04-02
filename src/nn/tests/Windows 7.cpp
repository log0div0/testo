
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 7/Установка") {
	{
		stb::Image image("Windows 7/Установка/Windows завершает применение параметров.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Настройка Windows").size() == 1);
		CHECK(tensor.match(&image, "Windows завершает применение параметров").size() == 1);
		CHECK(tensor.match(&image, "Справка").size() == 1);
		CHECK(tensor.match(&image, "Русский (Россия)").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите имя пользователя.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите имя пользователя для вашей учетной записи, а также имя компьютера в сети.").size() == 1);
		CHECK(tensor.match(&image, "Введите имя пользователя (например, Андрей):").size() == 1);
		CHECK(tensor.match(&image, "Введите имя компьютера:").size() == 1);
		CHECK(tensor.match(&image, "Корпорация Майкрософт (Microsoft Corp.), 2009. Все права защищены.").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите ключ продукта.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Введите ключ продукта Windows").size() == 1);
		CHECK(tensor.match(&image, "ключ продукта").size() == 2);
		CHECK(tensor.match(&image, "КЛЮЧ ПРОДУКТА").size() == 1);
		CHECK(tensor.match(&image, "Наклейка с ключом продукта выглядит так:").size() == 1);
		CHECK(tensor.match(&image, "XXXXX-XXXXX-XXXXX-XXXXX-XXXXX").size() == 1);
		CHECK(tensor.match(&image, "Автоматически активировать Windows при подключении к Интернету").size() == 1);
		CHECK(tensor.match(&image, "Что такое активация?").size() == 1);
		CHECK(tensor.match(&image, "Заявление о конфиденциальности").size() == 1);
		CHECK(tensor.match(&image, "Пропустить").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Введите пароль.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установите пароль для своей учетной записи").size() == 1);
		CHECK(tensor.match(&image, "Введите пароль (рекомендуется):").size() == 1);
		CHECK(tensor.match(&image, "Подтверждение пароля:").size() == 1);
		CHECK(tensor.match(&image, "Введите подсказку для пароля:").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выберите раздел для установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите раздел для установки Windows").size() == 1);
		CHECK(tensor.match(&image, "Файл").size() == 1);
		CHECK(tensor.match(&image, "Полный раз...").size() == 1);
		CHECK(tensor.match(&image, "Свободно").size() == 1);
		CHECK(tensor.match(&image, "Тип").size() == 1);
		CHECK(tensor.match(&image, "Незанятое место на диске 0").size() == 1);
		CHECK(tensor.match(&image, "20.0 ГБ").size() == 2);
		CHECK(tensor.match(&image, "Обновить").size() == 1);
		CHECK(tensor.match(&image, "Настройка диска").size() == 1);
		CHECK(tensor.match(&image, "Загрузка").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
		CHECK(tensor.match(&image, "Сбор информации").size() == 1);
		CHECK(tensor.match(&image, "Установка Windows").size() == 2);
	}
	{
		stb::Image image("Windows 7/Установка/Выберите тип установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите тип установки").size() == 1);
		CHECK(tensor.match(&image, "Обновление").size() == 3);
		CHECK(tensor.match(&image, "Полная установка (дополнительные параметры)").size() == 1);
		CHECK(tensor.match(&image, "Помощь в принятии решения").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор версии.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите операционную систему, которую следует установить").size() == 1);
		CHECK(tensor.match(&image, "Операционная система").size() == 1);
		CHECK(tensor.match(&image, "Архитектура").size() == 1);
		CHECK(tensor.match(&image, "Дата").size() == 1);
		CHECK(tensor.match(&image, "Windows 7 Начальная").size() == 2);
		CHECK(tensor.match(&image, "Windows 7 Домашняя базовая").size() == 2);
		CHECK(tensor.match(&image, "Windows 7 Домашняя расширенная").size() == 2);
		CHECK(tensor.match(&image, "Windows 7 Профессиональная").size() == 2);
		CHECK(tensor.match(&image, "Windows 7 Максимальная").size() == 2);
		CHECK(tensor.match(&image, "x86").size() == 5);
		CHECK(tensor.match(&image, "x64").size() == 4);
		CHECK(tensor.match(&image, "11/20/2010").size() == 5);
		CHECK(tensor.match(&image, "11/21/2010").size() == 4);
		CHECK(tensor.match(&image, "Описание:").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "My language is English").size() == 1);
		CHECK(tensor.match(&image, "Мой язык - русский").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Выбор языка 2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Устанавливаемый язык:").size() == 1);
		CHECK(tensor.match(&image, "Формат времени и денежных единиц:").size() == 1);
		CHECK(tensor.match(&image, "Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(tensor.match(&image, "Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Для продолжения требуется перезагрузка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Для продолжения требуется перезагрузка Windows").size() == 1);
		CHECK(tensor.match(&image, "Перезагрузка через 5 сек.").size() == 1);
		CHECK(tensor.match(&image, "Перезагрузить сейчас").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка даты и времени.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Проверьте настройку даты и времени").size() == 1);
		CHECK(tensor.match(&image, "Часовой пояс:").size() == 1);
		CHECK(tensor.match(&image, "Автоматический переход на летнее время и обратно").size() == 1);
		CHECK(tensor.match(&image, "Дата:").size() == 1);
		CHECK(tensor.match(&image, "Время:").size() == 1);
		CHECK(tensor.match(&image, "Ноябрь 2019").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Настройка сети.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите текущее место расположения компьютера").size() == 1);
		CHECK(tensor.match(&image, "Домашняя сеть").size() == 1);
		CHECK(tensor.match(&image, "Рабочая сеть").size() == 1);
		CHECK(tensor.match(&image, "Общественная сеть").size() == 1);
		CHECK(tensor.match(&image, "Если не уверены, выбирайте общественную сеть.").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ознакомьтесь с лицензией.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Ознакомьтесь с условиями лицензии").size() == 1);
		CHECK(tensor.match(&image, "УСЛОВИЯ ЛИЦЕНЗИИ НА ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ MICROSOFT").size() == 1);
		CHECK(tensor.match(&image, "WINDOWS 7 МАКСИМАЛЬНАЯ С ПАКЕТОМ ОБНОВЛЕНИЯ 1").size() == 1);
		CHECK(tensor.match(&image, "Я принимаю условия лицензии").size() == 1);
		CHECK(tensor.match(&image, "Далее").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Помогите автоматически защитить компьютер.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Помогите автоматически защитить компьютер и улучшить Windows").size() == 1);
		CHECK(tensor.match(&image, "Использовать рекомендуемые параметры").size() == 1);
		CHECK(tensor.match(&image, "Устанавливать только наиболее важные обновления").size() == 1);
		CHECK(tensor.match(&image, "Отложить решение").size() == 1);
		CHECK(tensor.match(&image, "Подробнее об этих параметрах").size() == 1);
		CHECK(tensor.match(&image, "Заявление о конфиденциальности").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Просто кнопка Установить.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установка Windows").size() == 1);
		CHECK(tensor.match(&image, "Windows 7").size() == 1);
		CHECK(tensor.match(&image, "Установить").size() == 1);
		CHECK(tensor.match(&image, "Что следует знать перед выполнением установки Windows").size() == 1);
		CHECK(tensor.match(&image, "Восстановление системы").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Рабочий стол сразу после установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Корзина").size() == 1);
		CHECK(tensor.match(&image, "RU").size() == 1);
		CHECK(tensor.match(&image, "13:56").size() == 1);
		CHECK(tensor.match(&image, "23.11.2019").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Ход установки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установка Windows...").size() == 1);
		CHECK(tensor.match(&image, "Копирование файлов Windows").size() == 1);
		CHECK(tensor.match(&image, "Распаковка файлов Windows (0%)").size() == 1);
		CHECK(tensor.match(&image, "Установка компонентов").size() == 1);
		CHECK(tensor.match(&image, "Установка обновлений").size() == 1);
		CHECK(tensor.match(&image, "Завершение установки").size() == 1);
	}
	{
		stb::Image image("Windows 7/Установка/Экран входа в систему.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "RU").size() == 1);
		CHECK(tensor.match(&image, "Петя").size() == 1);
		CHECK(tensor.match(&image, "Пароль").size() == 1);
		CHECK(tensor.match(&image, "Windows 7 Максимальная").size() == 1);
	}
}
