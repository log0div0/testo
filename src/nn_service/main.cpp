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
#include <ghc/filesystem.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "../js/Runtime.hpp"
#include "../nn/OnnxRuntime.hpp"

#include "MessageHandler.hpp"

namespace fs = ghc::filesystem;
using namespace std::chrono_literals;

#ifdef USE_CUDA
#include "GetDeviceInfo.hpp"
#include <license/License.hpp>

void verify_license(const std::string& path_to_license) {
	if (!fs::exists(path_to_license)) {
		throw std::runtime_error("File " + path_to_license + " does not exists");
	}

	std::string container = license::read_file(path_to_license);
	nlohmann::json license = license::unpack(container, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");

	license::Date not_before(license.at("not_before").get<std::string>());
	license::Date not_after(license.at("not_after").get<std::string>());
	license::Date now(std::chrono::system_clock::now());
	license::Date release_date(TESTO_RELEASE_DATE);

	if (now < release_date) {
		throw std::runtime_error("System time is incorrect");
	}

	if (now < not_before) {
		throw std::runtime_error("The license period has not yet come");
	}

	if (now > not_after) {
		throw std::runtime_error("The license period has already ended");
	}

	auto info = GetDeviceInfo(0);

	std::string device_uuid = license.at("device_uuid");
	if (info.uuid_str != device_uuid) {
		throw std::runtime_error("The graphics accelerator does not match the one specified in the license");
	}
}
#endif


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
			std::cout << "Error inside local acceptor loop: " << error.what() << std::endl;
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

		bool use_cpu = settings.value("use_cpu", false);

		#ifdef USE_CUDA
			if (!use_cpu) {
				if (!settings.count("license_path")) {
					throw std::runtime_error("To start the program you must specify the path to the license file (license_path in the settings file)");
				}
				verify_license(settings.at("license_path").get<std::string>());
			}

			nn::onnx::Runtime onnx_runtime(use_cpu);
		#else
			nn::onnx::Runtime onnx_runtime;
		#endif

		coro::Application(local_handler).run();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}