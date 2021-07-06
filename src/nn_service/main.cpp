#include <iostream>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <coro/Application.h>
#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>

#include <clipp.h>
#include <nlohmann/json.hpp>
#include "MessageHandler.hpp"

#include "../nn/OnnxRuntime.hpp"

using namespace std::chrono_literals;

nlohmann::json settings;

void local_handler() {
	auto port = settings.value("port", 8156);
	coro::TcpAcceptor acceptor(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port));
	std::cout << "Listening on port " << port << std::endl;
	acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		try {
			std::string new_connection = socket.handle().remote_endpoint().address().to_string() +
				":" + std::to_string(socket.handle().remote_endpoint().port());
			std::cout << "Accepted new connection: " << new_connection << std::endl;


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
		using namespace clipp;

		std::string settings_path;

		auto cli = clipp::group(
			value("path to settings", settings_path)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		{
			std::ifstream is(settings_path);
			if (!is) {
				throw std::runtime_error(std::string("Can't open settings file: ") + settings_path);
			}
			is >> settings;
		}

		nn::onnx::Runtime onnx_runtime;
		coro::Application(local_handler).run();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}