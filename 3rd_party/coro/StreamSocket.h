
#pragma once

#include "coro/Stream.h"
#include "coro/IoService.h"

namespace coro {

template <typename Protocol>
class StreamSocket: public Stream<typename Protocol::socket> {
public:
	using BaseType = Stream<typename Protocol::socket>;
	using BaseType::BaseType;
	using BaseType::operator=;
	using BaseType::_handle;

	StreamSocket(): BaseType(typename Protocol::socket(*IoService::current())) {
	}
	StreamSocket(const typename Protocol::endpoint& endpoint): BaseType(typename Protocol::socket(*IoService::current(), endpoint)) {
	}

	void connect(const typename Protocol::endpoint& endpoint) {
		AsioTask1 task;
		_handle.async_connect(endpoint, task.callback());
		task.wait(_handle);
	}

};

using TcpSocket = StreamSocket<asio::ip::tcp>;

}