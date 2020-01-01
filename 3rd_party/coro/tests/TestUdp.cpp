
#include "coro/DatagramSocket.h"
#include "coro/CoroPool.h"
#include <catch.hpp>

using namespace asio::ip;
using namespace coro;

TEST_CASE("A basic UDP socket test") {
	udp::endpoint serverEndpoint(address::from_string("127.0.0.1"), 44442), senderEndpoint;
	std::vector<uint8_t> testData { 0x01, 0x02, 0x03, 0x04 };

	DatagramSocket<udp> server(serverEndpoint);
	DatagramSocket<udp> client(udp::v4());

	client.send(asio::buffer(testData), serverEndpoint);

	{
		std::vector<uint8_t> tempData(4);
		server.receive(asio::buffer(tempData), senderEndpoint);
		REQUIRE(tempData == testData);
	}

	server.send(asio::buffer(testData), senderEndpoint);

	{
		std::vector<uint8_t> tempData(4);
		client.receive(asio::buffer(tempData), senderEndpoint);
		REQUIRE(tempData == testData);
		REQUIRE(serverEndpoint == senderEndpoint);
	}
}
