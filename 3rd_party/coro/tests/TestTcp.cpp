
#include "coro/Acceptor.h"
#include "coro/StreamSocket.h"
#include "coro/CoroPool.h"
#include <catch.hpp>

using namespace asio::ip;
using namespace coro;

TEST_CASE("TCP server + TCP client") {
	auto endpoint = tcp::endpoint(address::from_string("127.0.0.1"), 44442);
	std::vector<uint8_t> TestData { 0x01, 0x02, 0x03, 0x04 };

	bool serverDone = false, clientDone = false;

	CoroPool pool;
	pool.exec([&] {
		Acceptor<tcp> acceptor(endpoint);
		StreamSocket<tcp> socket = acceptor.accept();
		std::vector<uint8_t> data(4);
		REQUIRE(socket.read(asio::buffer(data)) == 4);
		REQUIRE(data == TestData);
		REQUIRE(socket.write(asio::buffer(data)) == 4);
		serverDone = true;
	});
	pool.exec([&] {
		StreamSocket<tcp> socket;
		socket.connect(endpoint);
		REQUIRE(socket.write(asio::buffer(TestData)) == 4);
		std::vector<uint8_t> data(4);
		REQUIRE(socket.read(asio::buffer(data)) == 4);
		REQUIRE(data == TestData);
		clientDone = true;
	});
	REQUIRE_NOTHROW(pool.waitAll(false));

	REQUIRE(serverDone);
	REQUIRE(clientDone);
}


TEST_CASE("Cancel Acceptor::accept") {
	auto endpoint = tcp::endpoint(address::from_string("127.0.0.1"), 44442);

	bool success = false;

	CoroPool pool;
	pool.exec([&] {
		Acceptor<tcp> acceptor(endpoint);
		try {
			acceptor.accept();
		}
		catch (const CancelError&) {
			success = true;
		}
	})->cancel();
	REQUIRE_NOTHROW(pool.waitAll(false));

	REQUIRE(success);
}