
#include "coro/IoService.h"
#ifdef _MSC_VER
#include "coro/FiberWindows.h"
#endif
#ifdef __GNUC__
#include "coro/FiberLinux.h"
#endif

namespace coro {

thread_local IoService* t_ioService = nullptr;

IoService* IoService::current() {
	if (!t_ioService) {
		throw std::runtime_error("IoService::current() is nullptr");
	}
	return t_ioService;
}

void IoService::run() {
	Fiber::initialize();
	t_ioService = this;
	while (_impl.run_one())
	{
	  while (_impl.poll_one());
	  while (checkpoints.size()) {
	  	auto checkpoint = std::move(checkpoints.front());
	  	checkpoints.pop();
	  	checkpoint();
	  }
	}
	t_ioService = nullptr;
	Fiber::deinitialize();
}

}