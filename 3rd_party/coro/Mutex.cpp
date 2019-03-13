
#include "coro/Mutex.h"
#include <coro/Finally.h>
#include <cassert>

namespace coro {

void Mutex::lock() {
	if (_owner) {
		Finally cleanup([&] {
			_coros.remove(Coro::current());
		});
		_coros.push_back(Coro::current());
		_coros.back()->yield({token(), TokenThrow});
	}

	assert(_owner == nullptr);
	_owner = Coro::current();
}

void Mutex::unlock() {
	assert(_owner == Coro::current());
	_owner = nullptr;

	if (!_coros.empty()) {
		_coros.front()->resume(token());
	}
}

std::string Mutex::token() const {
	return "Mutex " + std::to_string((uint64_t)this);
}

}