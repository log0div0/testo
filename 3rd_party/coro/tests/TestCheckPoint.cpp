
#include <catch.hpp>
#include "coro/CheckPoint.h"

using namespace coro;

TEST_CASE("CheckPoint") {
	CheckPoint();
	CheckPoint();
	CheckPoint();
	CheckPoint();
	CheckPoint();
}

