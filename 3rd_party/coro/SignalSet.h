
#pragma once

#include "coro/AsioTask.h"

namespace coro {

/// Wrapper вокруг asio::signal_set
class SignalSet {
public:
	SignalSet(const std::initializer_list<int32_t>& signals);

	int32_t wait();

private:
	asio::signal_set _handle;
};

}