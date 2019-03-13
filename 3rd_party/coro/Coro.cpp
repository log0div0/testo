
#include "coro/Coro.h"
#include <coro/Finally.h>
#include <algorithm>
#include <cassert>

#include <cxxabi.h>

using namespace __cxxabiv1;

namespace coro {

thread_local Coro* t_currentCoro = nullptr;

void
#ifdef _MSC_VER
__stdcall
#endif
Run(void* coro) {
	reinterpret_cast<Coro*>(coro)->run();
}

Coro* Coro::current() {
	if (!t_currentCoro) {
		throw std::logic_error("Coro::current is nullptr");
	}
	return t_currentCoro;
}

Coro::Coro(std::function<void()> routine): _routine(std::move(routine)), _fiber(Run, this),
	_previousCoro(nullptr), _tokens({TokenStart})
{
}

Coro::~Coro() {
	assert(_tokens.empty() || _tokens == std::vector<std::string>{TokenStart});
#ifdef _DEBUG
	std::string what;
	for (auto exception: _exceptions) {
		try {
			std::rethrow_exception(exception);
		}
		catch (const std::exception& error) {
			what += error.what();
			what += "\n";
		}
		catch (const CancelError&) {
			continue;
		}
		catch (...) {
			what += abi::__cxa_current_exception_type()->name();
			what += "\n";
		}
	}
	if (what.size()) {
		printf("Coro::~Coro: unhandled exceptions:\n%s", what.c_str());
	}
#endif
}

void Coro::start() {
	resume(TokenStart);
}

void Coro::resume(const std::string& token) {
	if (std::find(_tokens.begin(), _tokens.end(), token) == _tokens.end()) {
		return;
	}

	_previousCoro = t_currentCoro;
	t_currentCoro = this;
	if (_previousCoro) {
		_previousCoro->_fiber.switchTo(_fiber);
	} else {
		_fiber.enter();
	}
	t_currentCoro = _previousCoro;
	_previousCoro = nullptr;
}

void Coro::propagateException(std::exception_ptr exception) {
	assert(exception);
	_exceptions.push_back(exception);
	resume(TokenThrow);
}

void Coro::yield(std::vector<std::string> tokens) {
	_tokens = std::move(tokens);
	Finally clearTokens([&] {
		_tokens.clear();
	});

	if (std::find(_tokens.begin(), _tokens.end(), TokenThrow) != _tokens.end()) {
		propagateException();
	}

	if (_previousCoro) {
		_fiber.switchTo(_previousCoro->_fiber);
	} else {
		_fiber.exit();
	}

	if (std::find(_tokens.begin(), _tokens.end(), TokenThrow) != _tokens.end()) {
		propagateException();
	}
}

void Coro::cancel() {
	propagateException(CancelError());
}

void Coro::propagateException() {
	if (_exceptions.size()) {
		auto exception = _exceptions.front();
		_exceptions.pop_front();
		assert(exception);
		std::rethrow_exception(exception);
	}
}

void Coro::run() {
	try
	{
		_routine();
	}
	catch (const CancelError&) {
		// do nothing
	}
	catch (...)
	{
		auto exception = std::current_exception();
		assert(exception);
		_exceptions.push_front(exception);
	}
	_routine = nullptr;
	yield({});
}

}