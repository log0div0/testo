
#pragma once


#include "coro/Strand.h"
#include "coro/Coro.h"


namespace coro {

/// Базовый класс асинхронной операции
class AsioTask {
protected:
	/*!
		@brief Дожидается завершения асинхронной операции

		Если во время ожидания в корутину бросается исключение, то асинхронная операция отменяется
	*/
	template <typename Handle>
	void doWait(Handle& handle) {
		try {
			_coro->yield({token(), TokenThrow});
		}
		catch (...) {
			auto exception = std::current_exception();
			handle.cancel();
			_coro->yield({token()});
			assert(_isCallbackExecuted);

			// не используйте здесь throw, gcc это не переваривает
			std::rethrow_exception(exception);
		}
	}

	std::string token() const {
		return "AsioTask " + std::to_string((uint64_t)this);
	}

	Coro* _coro = Coro::current();
	bool _isCallbackExecuted = false;
};

/// Асинхронная операция без возвращаемого значения
class AsioTask1: public AsioTask {
public:
	/// Передайте этот callback в asio
	std::function<void(const std::error_code&)> callback() {
		return Strand::current()->wrap([=](const std::error_code& errorCode) {
			_isCallbackExecuted = true;
			_errorCode = errorCode;
			_coro->resume(token());
		});
	}

	/// @see AsioTask::doWait
	template <typename Handle>
	void wait(Handle& handle) {
		doWait(handle);

		if (_errorCode) {
			throw std::system_error(_errorCode);
		}
	}

private:
	std::error_code _errorCode;
};

/// Асинхронная операция c одним возвращаемым значением
template <typename Result>
class AsioTask2: public AsioTask {
public:
	/// Передайте этот callback в asio
	std::function<void(const std::error_code&, Result)> callback() {
		return Strand::current()->wrap([=](const std::error_code& errorCode, Result result) {
			_isCallbackExecuted = true;
			_errorCode = errorCode;
			_result = result;
			_coro->resume(token());
		});
	}

	/// @see AsioTask::doWait
	template <typename Handle>
	Result wait(Handle& handle) {
		doWait(handle);

		if (_errorCode) {
			throw std::system_error(_errorCode);
		}

		return _result;
	}

private:
	std::error_code _errorCode;
	Result _result;
};

}