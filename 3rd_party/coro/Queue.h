
#pragma once

#include "coro/Coro.h"
#include <coro/Finally.h>
#include <queue>
#include <list>

namespace coro {

/*!
	@brief Ещё один примитив синхронизации корутин

	@warning НЕ потокобезопасен!!!
*/
template <typename T>
class Queue {
public:
	/*!
		@brief Получить элемент из очереди

		Если очередь пуста, то происходит выход из корутины до тех пор пока очередь
		не наполнится. Или до тех пор, пока корутина не будет отменена.
	*/
	T pop() {
		if (_data.empty()) {
			Finally cleanup([&] {
				_coros.remove(Coro::current());
			});
			_coros.push_back(Coro::current());
			_coros.back()->yield({token(), TokenThrow});
		}

		T t = std::move(_data.front());
		_data.pop();
		return t;
	}

	/// Положить элемент в очередь
	template <typename U>
	void push(U&& u) {
		_data.push(std::forward<U>(u));

		if (!_coros.empty()) {
			_coros.front()->resume(token());
		}
	}

	size_t size() const {
		return _data.size();
	}

private:
	std::string token() const {
		return "Queue " + std::to_string((uint64_t)this);
	}

	std::queue<T> _data;
	std::list<Coro*> _coros;
};

}