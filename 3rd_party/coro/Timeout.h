
#pragma once

#include <asio/steady_timer.hpp>
#include <atomic>
#include "coro/Coro.h"
#include "coro/IoService.h"
#include "coro/Strand.h"

namespace coro {

class Timeout;

/// Исключение, выбрасываемое при срабатывании таймаута
class TimeoutError: public std::runtime_error {
public:
	TimeoutError(Timeout* timeout): std::runtime_error("Timeout was triggered"), _timeout(timeout) {}

	Timeout* timeout() const {
		return _timeout;
	}

private:
	Timeout* _timeout;
};

/*!
	@brief Таймаут, что ещё тут скажешь

	@warning Этот код НЕ РАБОТАЕТ:
	@code
		class A {
		public:
			enum { TIMEOUT = 10 };

			void f() {
				Timeout timeout(std::chrono::seconds(TIMEOUT));
				....
			}
		};
	@endcode
	Здесь не объявление переменной, здесь объявление ФУНКЦИИ
*/
class Timeout {
public:
	/// Установить таймаут
	template <typename Duration>
	Timeout(Duration duration): _timer(*IoService::current()) {
		_timer.expires_from_now(duration);
		_timer.async_wait(Strand::current()->wrap([=](const std::error_code& errorCode) {
			_callbackExecuted = true;
			if (_timerCanceled) {
				return _coro->resume(token());
			}
			if (errorCode) {
				return _coro->propagateException(std::system_error(errorCode));
			}
			_coro->propagateException(TimeoutError(this));
		}));
	}
	/// Снять таймаут
	~Timeout() {
		if (!_callbackExecuted) {
			_timerCanceled = true;
			_timer.cancel();
			waitCallbackExecution();
		}
	}

private:
	void waitCallbackExecution() {
		_coro->yield({token()});
	}


	std::string token() const {
		return "Timeout " + std::to_string((uint64_t)this);
	}

	asio::steady_timer _timer;
	Coro* _coro = Coro::current();
	bool _timerCanceled = false, _callbackExecuted = false;
};

}