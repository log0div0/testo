#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <coro/Application.h>
#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>

#include "MessageHandler.hpp"

#include "../nn/OnnxRuntime.hpp"

using namespace std::chrono_literals;

void local_handler() {
	coro::TcpAcceptor acceptor(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 8888));
	acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		try {
			std::cout << "Accepted\n";

			std::shared_ptr<Channel> channel(new Channel(std::move(socket)));

			MessageHandler message_handler(std::move(channel));
			message_handler.run();
		} catch (const std::exception& error) {
			std::cout << "Error inside local acceptor loop: " << error.what();
		}
	});
}

int main(int argc, char** argv) {
	try {
		nn::onnx::Runtime onnx_runtime;
		std::thread t1([]() {coro::Application(local_handler).run();});
		t1.join();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}