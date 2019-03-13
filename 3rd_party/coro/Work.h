
#pragma once

#include <coro/IoService.h>

namespace coro {

/// Wrapper вокруг asio::io_service::work
class Work {
public:

private:
	asio::io_service::work _impl = asio::io_service::work(*IoService::current());
};

}