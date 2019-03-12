
#pragma once

#include "coro/IoService.h"
#include "coro/AsioTask.h"

namespace coro {

/// Wrapper вокруг asio::ip::basic_resolver
template <typename InternetProtocol>
class Resolver {
public:
	typedef asio::ip::basic_resolver<InternetProtocol> Impl;
	typedef typename Impl::iterator Iterator;
	typedef typename Impl::query Query;

	Resolver(): _handle(*IoService::current()) {}

	Iterator resolve(const Query& query) {
		AsioTask2<Iterator> task;
		_handle.async_resolve(query, task.callback());
		return task.wait(_handle);
	}

private:
	Impl _handle;
};

typedef Resolver<asio::ip::udp> UdpResolver;
typedef Resolver<asio::ip::tcp> TcpResolver;

}