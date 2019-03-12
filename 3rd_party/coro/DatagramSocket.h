
#pragma once

#include "coro/AsioTask.h"
#include "coro/IoService.h"

namespace coro {

template <typename Protocol>
class DatagramSocket {
public:
	DatagramSocket(Protocol protocol = Protocol()): _handle(*IoService::current(), protocol) {

	}
	DatagramSocket(const typename Protocol::endpoint& endpoint): _handle(*IoService::current(), endpoint.protocol())
	{
		asio::socket_base::reuse_address option(true);
		_handle.set_option(option);
		_handle.bind(endpoint);
	}

	DatagramSocket(DatagramSocket&& other): _handle(std::move(other._handle)) {

	}

	DatagramSocket& operator=(DatagramSocket&& other) {
		_handle = std::move(other._handle);
		return *this;
	}

	template <typename T>
	size_t send(const T& t, const typename Protocol::endpoint& endpoint) {
		AsioTask2<size_t> task;
		_handle.async_send_to(t, endpoint, task.callback());
		return task.wait(_handle);
	}

	template <typename T>
	size_t receive(const T& t, typename Protocol::endpoint& endpoint) {
		AsioTask2<size_t> task;
		_handle.async_receive_from(t, endpoint, task.callback());
		return task.wait(_handle);
	}

	const typename Protocol::socket& handle() const {
		return _handle;
	}

protected:
	typename Protocol::socket _handle;
};

using UdpSocket = DatagramSocket<asio::ip::udp>;

}