
#pragma once

#include "coro/IoService.h"
#include "coro/Coro.h"

/*!
 	@brief Библиотека для работы с асинхронным вводом/выводом с синхронным кодом
 	@see @ref md_Docs_Про_корутины
*/
namespace coro {

/*!
	@brief Используйте этот класс, для того, чтобы создать приложение на основе корутин

	Пример:
	@code
	void main() {
		coro::Application([&] {
			// Здесь можно пользоваться корутинами
		}).run();
	}
	@endcode
*/
class Application {
public:
	Application(const std::function<void()>& main);
	/// Дожидается завершения всех корутин (НЕ отменяет их)
	~Application();

	/// Запускает приложение в текущем потоке
	void run();
	/// Отменяет корневую корутину (планирует выброс исключения) и сразу возвращает управление
	void cancel();

private:
	IoService _ioService;
	Coro _coro;
};

}