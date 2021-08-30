
#include <iostream>
#include <chrono>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <tchar.h>

#include <winapi/Functions.hpp>

#include <coro/Application.h>
#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>


#include <nlohmann/json.hpp>
#include <ghc/filesystem.hpp>

#include "../nn/OnnxRuntime.hpp"
#include "MessageHandler.hpp"

namespace fs = ghc::filesystem;

#include "../license/GetDeviceInfo.hpp"
#include <license/License.hpp>

#ifdef USE_CUDA
void verify_license(const std::string& path_to_license) {
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

void app_main() {
	try {
		bool use_gpu = settings.value("use_gpu", false);

		if (use_gpu) {
			if (!settings.count("license_path")) {
				throw std::runtime_error("To start the program in GPU mode you must specify the path to the license file (license_path in the settings file)");
			}
#ifdef USE_CUDA
			spdlog::info("Verifying license...");
			verify_license(settings.at("license_path").get<std::string>());
			spdlog::info("License is OK");
#endif
		}

		nn::onnx::Runtime onnx_runtime(!use_gpu);

		spdlog::info("Starting testo nn service");
		spdlog::info("Testo framework version: {}", TESTO_VERSION);
		spdlog::info("GPU mode enabled: {}", use_gpu);

		local_handler();
	} catch (const std::exception& err) {
		spdlog::error("app_main std error: {}", err.what());
	} catch (const coro::CancelError&) {
		spdlog::error("app_main CancelError");
	} catch (...) {
		spdlog::error("app_main unknown error");
	}
};

coro::Application app(app_main);

void StopApp() {
	app.cancel();
}


#define SERVICE_NAME _T("Testo NN Service")

void ControlHandler(DWORD request) {
	switch(request)
	{
	case SERVICE_CONTROL_STOP:
		spdlog::info("SERVICE_CONTROL_STOP BEGIN");
		StopApp();
		spdlog::info("SERVICE_CONTROL_STOP END");
		break;
	case SERVICE_CONTROL_SHUTDOWN:
		spdlog::info("SERVICE_CONTROL_SHUTDOWN BEGIN");
		StopApp();
		spdlog::info("SERVICE_CONTROL_SHUTDOWN END");
		break;
	default:
		break;
	}
}

void ServiceMain(int argc, char** argv) {
	SERVICE_STATUS serviceStatus = {};
	serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
	serviceStatus.dwCurrentState = SERVICE_START_PENDING;
	serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN;
	serviceStatus.dwWin32ExitCode = 0;
	serviceStatus.dwServiceSpecificExitCode = 0;
	serviceStatus.dwCheckPoint = 0;
	serviceStatus.dwWaitHint = 0;

	SERVICE_STATUS_HANDLE serviceStatusHandle = RegisterServiceCtrlHandler(SERVICE_NAME, (LPHANDLER_FUNCTION)ControlHandler);
	if (!serviceStatusHandle) {
		throw std::runtime_error("RegisterServiceCtrlHandler failed");
	}

	spdlog::info("NN service start");
	serviceStatus.dwCurrentState = SERVICE_RUNNING;
	SetServiceStatus(serviceStatusHandle, &serviceStatus);
	
	app.run();
	
	spdlog::info("NN service stop");
	serviceStatus.dwCurrentState = SERVICE_STOPPED;
	SetServiceStatus(serviceStatusHandle, &serviceStatus);
}

void setup_logs(const fs::path& log_file_path) {
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

int _tmain(int argc, TCHAR *argv[]) {
	fs::path parent_folder = fs::path(winapi::get_module_file_name()).parent_path().parent_path();
	fs::path settings_path = parent_folder / "nn_service.json";
	fs::path logs_path = parent_folder / "nn_service_logs.txt";

	try {
		std::ifstream is(settings_path.generic_string());
		if (!is) {
			throw std::runtime_error(std::string("Can't open settings file: ") + settings_path.generic_string());
		}
		is >> settings;
		setup_logs(logs_path);
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}

	try {
		SERVICE_TABLE_ENTRY ServiceTable[] =
		{
			{SERVICE_NAME, (LPSERVICE_MAIN_FUNCTION) ServiceMain},
			{NULL, NULL}
		};

		if (StartServiceCtrlDispatcher(ServiceTable) == FALSE) {
			throw std::runtime_error("StartServiceCtrlDispatcher failed");
		}
	}
	catch (const std::exception& error) {
		spdlog::error("Error in main function");
		spdlog::error(error.what());
		return -1;
	}
}
