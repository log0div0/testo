
#pragma once

#include <asio/windows/object_handle.hpp>
#include "coro/AsioTask.h"
#include "coro/IoService.h"

namespace coro {

class WindowsObjectHandle {
public:
	WindowsObjectHandle(): _handle(*IoService::current()) {}
	WindowsObjectHandle(HANDLE handle): _handle(*IoService::current(), handle) {}

	void wait() {
		AsioTask1 task;
		_handle.async_wait(task.callback());
		task.wait(_handle);
	}

	asio::windows::object_handle& handle() {
		return _handle;
	}

private:
	asio::windows::object_handle _handle;
};

}