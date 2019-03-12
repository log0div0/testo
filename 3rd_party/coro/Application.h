
#pragma once

#include "coro/IoService.h"
#include "coro/Strand.h"
#include "coro/Coro.h"
#include <thread>

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
		}).runAsync();
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
	/// Запускает приложение threadCount потоках и сразу возвращает управление
	void runAsync(size_t threadsCount = std::thread::hardware_concurrency());
	/// Отменяет корневую корутину (планирует выброс исключения) и сразу возвращает управление
	void cancel();
	/// Бросить исключение в корневую корутину
	void propagateException(std::exception_ptr exception);
	/// Бросить исключение в корневую корутину
	template <typename Exception>
	void propagateException(Exception exception) {
		propagateException(std::make_exception_ptr(exception));
	}

private:
	IoService _ioService;
	Strand _strand;
	Coro _coro;
	std::vector<std::thread> _threads;
};

}