
#pragma once

#include <iostream>
#include <chrono>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <coro/Application.h>
#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>

#include <nlohmann/json.hpp>
#include <ghc/filesystem.hpp>

#include "nn/OnnxRuntime.hpp"
#include "MessageHandler.hpp"

namespace fs = ghc::filesystem;

#include "license/GetDeviceInfo.hpp"
#include "license/License.hpp"

#ifdef USE_CUDA
void verify_license(const nlohmann::json& settings) {
	std::string path_to_license = settings.at("license_path").get<std::string>();

	if (!fs::exists(path_to_license)) {
		throw std::runtime_error("File " + path_to_license + " does not exist");
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

void local_handler(const nlohmann::json& settings) {
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

void setup_logs(const nlohmann::json& settings) {
	std::string log_file_path = settings.at("log_file").get<std::string>();

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path);
	auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
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

nlohmann::json load_settings(const std::string& settings_path) {
	nlohmann::json settings;
	if (!fs::exists(settings_path)) {
		fs::create_directories(fs::path(settings_path).parent_path());
		std::ofstream os(settings_path);
		if (!os) {
			throw std::runtime_error(std::string("Can't open settings file: ") + settings_path);
		}
		os << R"(
{
	"port": 8156,
	"log_level": "info",
	"license_path": "/opt/testo_license.lic",
	"use_gpu": false,
	"gpu_id": 0
}
)";
	}
	std::ifstream is(settings_path);
	if (!is) {
		throw std::runtime_error(std::string("Can't open settings file: ") + settings_path);
	}
	is >> settings;
	return settings;
}

void app_main(const nlohmann::json& settings) {
	try {
		setup_logs(settings);

		bool use_gpu = settings.value("use_gpu", false);
		size_t gpu_id = settings.value("gpu_id", 0);

		if (use_gpu) {
			if (!settings.count("license_path")) {
				throw std::runtime_error("To start the program in GPU mode you must specify the path to the license file (license_path in the settings file)");
			}
#ifdef USE_CUDA
			spdlog::info("Verifying license...");
			verify_license(settings);
			spdlog::info("License is OK");
#endif
		} else if (settings.count("gpu_id")) {
			spdlog::info("Ignoring 'gpu_id' setting because GPU mode is disabled...");
		}

		nn::onnx::Runtime onnx_runtime(!use_gpu, gpu_id);

		spdlog::info("Starting testo nn server");
		spdlog::info("Testo framework version: {}", TESTO_VERSION);
		spdlog::info("GPU mode enabled: {}", use_gpu);
		local_handler(settings);
	} catch (const std::exception& error) {
		spdlog::error(error.what());
	} catch (...) {
		spdlog::error("Unknown exception");
	}
}
