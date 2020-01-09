
#include "coro/Queue.h"
#include <catch.hpp>

using namespace coro;

TEST_CASE("A basic queue test") {
	std::vector<uint32_t> actual, expected = {0, 1, 2, 3};

	Queue<uint32_t> queue;

	Coro consumer([&] {
		for (uint32_t i = 0; i < 4; ++i) {
			actual.push_back(queue.pop());
		}
	});
	Coro producer([&] {
		for (uint32_t i = 0; i < 4; ++i) {
			queue.push(i);
		}
	});
	consumer.start();
	producer.start();

	REQUIRE(actual == expected);
}


TEST_CASE("Throw exception from Queue::pop") {
	std::vector<uint32_t> actual, expected = {};

	Queue<uint32_t> queue;

	Coro consumer([&] {
		actual.push_back(queue.pop());
	});
	Coro producer([&] {
		queue.push(0);
	});
	consumer.start();
	consumer.cancel();
	producer.start();

	REQUIRE(actual == expected);
}
