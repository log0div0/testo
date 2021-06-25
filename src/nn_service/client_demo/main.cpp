#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <coro/Application.h>
#include <coro/StreamSocket.h>

#include "../Channel.hpp"

using namespace std::chrono_literals;

void local_handler() {
	try {
		auto endpoint = Endpoint(asio::ip::address::from_string("127.0.0.1"), 8888);
		Socket socket;
		socket.connect(endpoint);
		std::cout << "Connected\n";

		std::shared_ptr<Channel> channel(new TCPChannel(std::move(socket)));

		Request msg(stb::Image<stb::RGB>("tmp.png"), std::string("Hello"));
		channel->send_request(msg);
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	

	/*acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		try {
			std::cout << "Accepted\n";

			std::shared_ptr<Channel> channel(new TCPChannel(std::move(socket)));
			while (true) {
				auto screenshot = channel->receive();
				std::cout << "Received screenshot\n";
			}
			//MessageHandler message_handler(std::move(channel));
			//message_handler.run();
		} catch (const std::exception& error) {
			std::cout << "Error inside local acceptor loop: " << error.what();
		}
	});*/
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