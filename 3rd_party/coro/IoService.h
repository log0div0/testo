
#pragma once

#include <asio.hpp>
#include <queue>

namespace coro {

/// Wrapper вокруг asio::io_service
class IoService {
public:
	static IoService* current();

	void run();

	template <typename T>
	void post(T&& t) {
		_impl.post(std::forward<T>(t));
	}

	template <typename T>
	void dispatch(T&& t) {
		_impl.dispatch(std::forward<T>(t));
	}

	std::queue<std::function<void()>> checkpoints;

	operator const asio::io_service&() const { return _impl; }
	operator asio::io_service&() { return _impl; }

private:
	asio::io_service _impl;
};

}