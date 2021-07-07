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

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "../js/Runtime.hpp"

#include "MessageHandler.hpp"

#include "../nn/OnnxRuntime.hpp"

using namespace std::chrono_literals;

nlohmann::json settings;

void local_handler() {
	auto port = settings.value("port", 8156);
	coro::TcpAcceptor acceptor(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port));
	spdlog::info(fmt::format("Listening on port {}", port));
	acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		std::string new_connection;
		try {
			new_connection = socket.handle().remote_endpoint().address().to_string() +
				":" + std::to_string(socket.handle().remote_endpoint().port());
			spdlog::info(fmt::format("Accepted new connection: {}", new_connection));

			std::shared_ptr<Channel> channel(new Channel(std::move(socket)));

			MessageHandler message_handler(std::move(channel));
			message_handler.run();
		} catch (const std::system_error& error) {
			if (error.code().value() == 2) {
				spdlog::info(fmt::format("Connection broken: {}", new_connection));
			}
		} catch (const std::exception& error) {
			std::cout << "Error inside local acceptor loop: " << error.what();
		}
	});
}

void setup_logs() {
	auto log_file_path = settings.value("log_file", "/var/log/testo_nn_service.log");
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path);
	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});

	std::string log_level = settings.value("log_level", "info");
	if (log_level == "info") {
		logger->set_level(spdlog::level::info);
		logger->flush_on(spdlog::level::info);
	} else if (log_level == "trace") {
		logger->set_level(spdlog::level::trace);
		logger->flush_on(spdlog::level::trace);
	} else {
		throw std::runtime_error("Only \"info\" and \"trace\" log levels are supported");
	}
	spdlog::set_default_logger(logger);
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

		setup_logs();

		nn::onnx::Runtime onnx_runtime;

		coro::Application(local_handler).run();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}