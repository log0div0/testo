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

#include "../license/GetDeviceInfo.hpp"
#include <license/License.hpp>

#ifdef USE_CUDA
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

#define APP_NAME "testo_nn_service"
#define PID_FILE_PATH ("/var/run/" APP_NAME ".pid")
#define LOG_FILE_PATH ("/var/log/" APP_NAME ".log")

struct pid_file_t {
	pid_file_t(pid_t pid) {
		std::ofstream stream(PID_FILE_PATH);
		if (!stream) {
			throw std::runtime_error("Creating pid file failure");
		}
		stream << pid << std::endl;
		stream.flush();
	}
	~pid_file_t() {
		unlink(PID_FILE_PATH);
	}
};

pid_t get_pid() {
	std::ifstream stream(PID_FILE_PATH);

	if (!stream) {
		return 0;
	}

	pid_t pid;
	stream >> pid;
	return pid;
}

struct cmdline: std::vector<std::string> {
	cmdline(pid_t pid) {
		std::string filename = "/proc/" + std::to_string(pid) + "/cmdline";
		std::ifstream stream(filename);
		if (!stream) {
			return;
		}
		for (std::string arg; std::getline(stream, arg, '\0'); ) {
			push_back(arg);
		}
	}
};

inline std::ostream& operator<<(std::ostream& stream, const cmdline& cmdline) {
	for (size_t i = 0; i < cmdline.size(); ++i) {
		if (i) {
			stream << " ";
		}
		stream << cmdline[i];
	}
	return stream;
}

bool is_running() {
	auto pid = get_pid();

	if (!pid) {
		return false;
	}

	if (kill(pid, 0) < 0) {
		return false;
	}
	else {
		cmdline cmd(pid);
		try {
			if (cmd.at(0).find(APP_NAME) != std::string::npos) {
				return true;
			} else {
				return false;
			}
		}
		catch (const std::out_of_range& error) {
			return false;
		}
	}
}

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
	auto log_file_path = settings.value("log_file", LOG_FILE_PATH);
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

enum class mode {
	start,
	stop,
	status,
	help
};

struct StartArgs {
	std::string settings_path = "/etc/testo/nn_service.json";
	bool foreground_mode = false; 
};

void app_main(const std::string& settings_path) {
	try {
		std::ifstream is(settings_path);
		if (!is) {
			throw std::runtime_error(std::string("Can't open settings file: ") + settings_path);
		}
		is >> settings;
		setup_logs();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}

	try {
		bool use_gpu = settings.value("use_gpu", false);

		if (use_gpu) {
			if (!settings.count("license_path")) {
				throw std::runtime_error("To start the program in GPU mode you must specify the path to the license file (license_path in the settings file)");
			}
#ifdef USE_CUDA
			verify_license(settings.at("license_path").get<std::string>());
#endif
		}

		nn::onnx::Runtime onnx_runtime(!use_gpu);

		spdlog::info("Starting testo nn service");
		spdlog::info("Testo framework version: {}", TESTO_VERSION);
		spdlog::info("GPU mode enabled: {}", use_gpu);
		coro::Application(local_handler).run();
	} catch (const std::exception& error) {
		spdlog::error(error.what());
	}
}

void start(const StartArgs& args) {
	if (!args.foreground_mode) {
		if (daemon(1, 0) < 0) {
			throw std::system_error(errno, std::system_category());
		}
	}
	auto pid  = getpid();
	pid_file_t pid_t(pid); 

	app_main(args.settings_path);
}

void stop() {
	auto pid = get_pid();
	if (!pid) {
		return;
	}

	if (kill(pid, SIGTERM) < 0) {
		throw std::system_error(errno, std::system_category());
	}
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		StartArgs start_args;
		mode selected_mode;

		auto start_spec = (
			command("start").set(selected_mode, mode::start),
			(option("--settings_file") & value("path to file", start_args.settings_path)) % "Path to settings file (default is /etc/testo/nn_service.json",
			(option("--foreground").set(start_args.foreground_mode)) % "Run in foreground"
		);

		auto cli = (
			start_spec |
			command("stop").set(selected_mode, mode::stop) |
			command("status").set(selected_mode, mode::status) |
			command("help").set(selected_mode, mode::help)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		switch (selected_mode) {
			case mode::start:
				start(start_args);
				return 0;
			case mode::stop:
				stop();
				return 0;
			case mode::status:
				if (is_running()) {
					std::cout << "RUNNING" << std::endl;
					return 1;
				} else {
					std::cout << "STOPPED" << std::endl;
					return 0;
				}
				return 0;
			case mode::help:
				std::cout << make_man_page(cli, APP_NAME) << std::endl;
				return 0;
			default:
				throw std::runtime_error("Unknown mode");
		}
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
}