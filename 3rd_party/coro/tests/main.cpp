
#include "coro/Application.h"
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main( int argc, char* const argv[] ) {
	int result;
	coro::Application([&] {
		result = Catch::Session().run(argc, argv);
	}).run();
	return result;
}
