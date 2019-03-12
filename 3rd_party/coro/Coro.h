
#pragma once

#ifdef _MSC_VER
#include "coro/FiberWindows.h"
#endif
#ifdef __GNUC__
#include "coro/FiberLinux.h"
#endif
#include <functional>
#include <string>
#include <list>
#include <vector>

namespace coro {

static constexpr auto TokenStart = "__start__";
static constexpr auto TokenThrow = "__throw__";

/*!
	@brief Икслючение для отмены корутин

	Специально не наследуется от std::exception, для того чтобы гарантированно полностью раскрутить
	стек корутины. Помни об этом, когда будешь писасть catch (...)
*/
struct CancelError {};

/// Корутина сферическая в вакууме
class Coro {
public:
	static Coro* current();

	Coro(std::function<void()> routine);
	~Coro();

	Coro(const Coro& other) = delete;
	Coro(Coro&& other);

	Coro& operator=(const Coro& other) = delete;
	Coro& operator=(Coro&& other);

	void start();
	/*!
		@brief Вход в корутину

		Может быть вызван как извне корутины, так и из другой корутины.

		@warning
			Избегайте циклического вызова корутин. Например:
			coro1 -> coro2 -> coro1.
			При необходимости используйте отложенный вызов корутины с помощью Strand::post
	*/
	void resume(const std::string& token);

	/*!
		@brief Выход из корутины

		Может быть вызван ТОЛЬКО из корутины из которой осуществляется выход. Вот так:

		@code
			Coro::current()->yield();
		@endcode
	*/
	void yield(std::vector<std::string> tokens);

	/// Бросить исключение в корутину
	void propagateException(std::exception_ptr exception);
	/// Бросить исключение в корутину
	template <typename Exception>
	void propagateException(Exception exception) {
		propagateException(std::make_exception_ptr(exception));
	}
	void propagateException();
	/// Бросить в корутину исключение CancelError
	void cancel();

	/// Очередь запланированных исключений
	const std::list<std::exception_ptr>& exceptions() const {
		return _exceptions;
	}

private:
	std::function<void()> _routine;
	Fiber _fiber;
	Coro* _previousCoro;
	std::list<std::exception_ptr> _exceptions;
	std::vector<std::string> _tokens;

public:
	void run();
};

}