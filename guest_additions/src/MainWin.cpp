
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <tchar.h>

#include <winapi/Functions.hpp>
#include <coro/Application.h>

#include "MessageHandler.hpp"

MessageHandler* g_message_handler = nullptr;

std::function app_main = [&]() {
	try {
		MessageHandler message_handler;
		g_message_handler = &message_handler;
		message_handler.run();
	} catch (const std::exception& err) {
		spdlog::error("app_main std error: {}", err.what());
	} catch (const coro::CancelError&) {
		spdlog::error("app_main CancelError");
	} catch (...) {
		spdlog::error("app_main unknown error");
	}
};

coro::Application app(app_main);

#define SERVICE_NAME _T("Testo Guest Additions")

void ControlHandler(DWORD request) {
	switch(request)
	{
	case SERVICE_CONTROL_STOP:
		spdlog::info("SERVICE_CONTROL_STOP BEGIN");
		g_message_handler->force_cancel();
		spdlog::info("SERVICE_CONTROL_STOP END");
		break;
	case SERVICE_CONTROL_SHUTDOWN:
		spdlog::info("SERVICE_CONTROL_SHUTDOWN BEGIN");
		g_message_handler->force_cancel();
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

	spdlog::info("App start");
	serviceStatus.dwCurrentState = SERVICE_RUNNING;
	SetServiceStatus(serviceStatusHandle, &serviceStatus);
	app.run();
	spdlog::info("App stop");
	serviceStatus.dwCurrentState = SERVICE_STOPPED;
	SetServiceStatus(serviceStatusHandle, &serviceStatus);
}

int _tmain (int argc, TCHAR *argv[]) {
	fs::path path = fs::path(winapi::get_module_file_name()).replace_extension("txt");
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.generic_string());
	auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	try {
		spdlog::info("Started");

		SERVICE_TABLE_ENTRY ServiceTable[] =
		{
			{SERVICE_NAME, (LPSERVICE_MAIN_FUNCTION) ServiceMain},
			{NULL, NULL}
		};

		if (StartServiceCtrlDispatcher(ServiceTable) == FALSE) {
			throw std::runtime_error("StartServiceCtrlDispatcher failed");
		}

		spdlog::info("Stopped");
	}
	catch (const std::exception& error) {
		spdlog::error("Error in main function");
		spdlog::error(error.what());
		return -1;
	}
}
