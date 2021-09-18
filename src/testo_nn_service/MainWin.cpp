
#include "Main.hpp"
#include <tchar.h>
#include <winapi/Functions.hpp>

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
