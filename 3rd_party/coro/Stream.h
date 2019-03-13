
#pragma once


#include "coro/AsioTask.h"


namespace coro {

template <typename Handle>
class Stream {
public:
	Stream(Handle handle): _handle(std::move(handle)) {}

	Stream(Stream&& other): _handle(std::move(other._handle)) {}

	Stream& operator=(Stream&& other) {
		_handle = std::move(other._handle);
		return *this;
	}

	template <typename ...T>
	size_t write(T&&... t) {
		AsioTask2<size_t> task;
		asio::async_write(_handle, asio::buffer(std::forward<T>(t)...), task.callback());
		return task.wait(_handle);
	}

	template <typename ...T>
	size_t read(T&&... t) {
		AsioTask2<size_t> task;
		asio::async_read(_handle, asio::buffer(std::forward<T>(t)...), task.callback());
		return task.wait(_handle);
	}

	template <typename ...T>
	size_t writeSome(T&&... t) {
		AsioTask2<size_t> task;
		_handle.async_write_some(asio::buffer(std::forward<T>(t)...), task.callback());
		return task.wait(_handle);
	}

	template <typename ...T>
	size_t readSome(T&&... t) {
		AsioTask2<size_t> task;
		_handle.async_read_some(asio::buffer(std::forward<T>(t)...), task.callback());
		return task.wait(_handle);
	}

	Handle& handle() {
		return _handle;
	}

	const Handle& handle() const {
		return _handle;
	}

protected:
	Handle _handle;
};

}