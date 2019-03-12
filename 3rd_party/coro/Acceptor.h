
#pragma once

#include "coro/AsioTask.h"
#include "coro/IoService.h"
#include "coro/CoroPool.h"

namespace coro {

/// Wrapper вокруг asio::ip::tcp::acceptor
template <typename Protocol>
class Acceptor {
public:
	Acceptor(const typename Protocol::endpoint& endpoint): _handle(*IoService::current())
	{
		_handle.open(endpoint.protocol());
		asio::socket_base::reuse_address option(true);
		_handle.set_option(option);
		_handle.bind(endpoint);
		_handle.listen();
	}

	typename Protocol::socket accept()
	{
		typename Protocol::socket socket(*IoService::current());

		AsioTask1 task;
		_handle.async_accept(socket, task.callback());
		task.wait(_handle);

		return socket;
	}

	/*!
		@brief В цикле принимает подключения и запускает их обработчики в отдельных Strand
		@param callback - функция-обработчик соединения
	*/
	void run(std::function<void(typename Protocol::socket)> callback)
	{
		CoroPool coroPool;
		while (true) {
			auto socket = accept();
			coroPool.exec([&] {
				callback(std::move(socket));
			});
		}
	}

	typename Protocol::acceptor& handle() {
		return _handle;
	}

protected:
	typename Protocol::acceptor _handle;
};

using TcpAcceptor = Acceptor<asio::ip::tcp>;

}
