#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <coro/Application.h>
#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>

#include "Channel.hpp"

using namespace std::chrono_literals;

void local_handler() {
	std::cout << "Local local_handler\n";
	coro::TcpAcceptor acceptor(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 8888));
	acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		try {
			std::cout << "Accepted\n";

			std::shared_ptr<Channel> channel(new TCPChannel(std::move(socket)));
			while (true) {
				auto message = channel->receive_request();
				std::cout << "Received message\n";
				message.screenshot.write_png("suchka.png");
			}
			//MessageHandler message_handler(std::move(channel));
			//message_handler.run();
		} catch (const std::exception& error) {
			std::cout << "Error inside local acceptor loop: " << error.what();
		}
	});
}

int main(int argc, char** argv) {
	try {
		std::thread t1([]() {coro::Application(local_handler).run();});
		t1.join();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}