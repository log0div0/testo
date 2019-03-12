
#include "coro/SignalSet.h"
#include "coro/IoService.h"

namespace coro {

SignalSet::SignalSet(const std::initializer_list<int32_t>& signals)
	: _handle(*IoService::current())
{
	for (auto signal: signals) {
		_handle.add(signal);
	}
}

int32_t SignalSet::wait() {
	AsioTask2<int32_t> task;
	_handle.async_wait(task.callback());
	return task.wait(_handle);
}

}