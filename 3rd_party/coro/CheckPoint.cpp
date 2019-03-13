
#include "coro/CheckPoint.h"
#include "coro/Coro.h"
#include "coro/Strand.h"

namespace coro {

void CheckPoint() {
	auto coro = Coro::current();
	std::string CheckPointToken = "CheckPoint " + std::to_string((uint64_t)coro);
	Strand::current()->post([=] {
		coro->resume(CheckPointToken);
	});
	coro->yield({CheckPointToken, TokenThrow});
}

}