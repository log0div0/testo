
#include <coro/Finally.h>
#include <catch.hpp>
#include "coro/Coro.h"

using namespace coro;

TEST_CASE("A basic test", "[Coro]") {
	bool success = false;
	Coro coro([&] {
		success = true;
	});
	coro.start();
	REQUIRE(success);
}

TEST_CASE("Nested coros", "[Coro]") {
	bool success = false;
	Coro coro1([&] {
		Coro coro2([&] {
			success = true;
		});
		auto current = Coro::current();
		coro2.start();
		REQUIRE(current == Coro::current());
	});
	coro1.start();
	REQUIRE(success);
}

TEST_CASE("Cancellation", "[Coro]") {
	bool success = false;
	Coro coro([&] {
		try {
			Coro::current()->yield({TokenThrow});
		}
		catch (const CancelError&) {
			success = true;
		}
	});
	coro.start();
	coro.cancel();
	REQUIRE(success);
}


TEST_CASE("Throw an exception into a coro", "[Coro]") {
	bool success = false;
	Coro coro([&] {
		try {
			Coro::current()->yield({TokenThrow});
		}
		catch (...) {
			success = true;
		}
	});
	coro.start();
	coro.propagateException(std::runtime_error("test"));
	REQUIRE(success);
}

TEST_CASE("MSVC bug: std::current_exception() == nullptr, if there are >=2 exceptions thrown", "[.]") {
	try {
		Finally throwInner([&] {
			try {
				throw std::runtime_error("inner");
			}
			catch (...) {
				REQUIRE(std::current_exception());
			}
		});
		throw std::runtime_error("outter");
	}
	catch(...) {
		REQUIRE(std::current_exception());
	}
}

TEST_CASE("Ensure that std::current_exception() != nullptr if exceptions are thrown in separete coros", "[Coro]") {
	Coro coro1([] {
		Coro coro2([] {
			try {
				throw std::runtime_error("inner");
			}
			catch (...) {
				REQUIRE(std::current_exception());
			}
		});
		try {
			Finally throwInner([&] {
				coro2.start();
			});
			throw std::runtime_error("outter");
		}
		catch (...) {
			REQUIRE(std::current_exception());
		}
	});
	coro1.start();
}
