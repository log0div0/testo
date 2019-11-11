
#include "coro/IoService.h"
#include "coro/CheckPoint.h"
#include "coro/Coro.h"

namespace coro {

void CheckPoint() {
	auto coro = Coro::current();
	std::string CheckPointToken = "CheckPoint " + std::to_string((uint64_t)coro);
	IoService::current()->post([=] {
		coro->resume(CheckPointToken);
	});
	coro->yield({CheckPointToken, TokenThrow});
}

}