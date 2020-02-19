
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <locale>
#include <codecvt>
#include <chrono>

using namespace std::chrono_literals;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <clipp.h>

#include "process/Process.hpp"

#include <shellapi.h>

namespace fs = std::filesystem;

enum class Command {
	Install,
	ShowHelp
};

#define APP_NAME "testo-guest-additions-helper"

std::string targetdir;
Command selected_command;

struct Service {
	Service(SC_HANDLE handle_): handle(handle_) {

	}

	~Service() {
		if (handle) {
			CloseServiceHandle(handle);
		}
	}

	Service(Service&& other);
	Service& operator=(Service&& other);

	void start() {
		if (!StartService(handle, 0, NULL)) {
			throw std::runtime_error("StartService failed");
		}
	}

	SERVICE_STATUS queryStatus() {
		SERVICE_STATUS status = {};
		if (!QueryServiceStatus(handle, &status)) {
			throw std::runtime_error("QueryServiceStatus failed");
		}
		return status;
	}

private:
	SC_HANDLE handle = NULL;
};

struct SCManager {
	SCManager() {
		handle = OpenSCManager(NULL, NULL, SC_MANAGER_CREATE_SERVICE);
		if (!handle) {
			throw std::runtime_error("OpenSCManager failed");
		}
	}

	~SCManager() {
		if (handle) {
			CloseServiceHandle(handle);
		}
	}

	SCManager(SCManager&& other);
	SCManager& operator=(SCManager&& other);

	Service service(const std::string& name) {
		SC_HANDLE hService = OpenServiceA(handle, name.c_str(), SERVICE_QUERY_STATUS | SERVICE_START);
		if (!hService) {
			throw std::runtime_error("OpenServiceA failed");
		}
		return hService;
	}

private:
	SC_HANDLE handle = NULL;
};

void install() {
	spdlog::info("Install ...");

	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);

	{
		fs::path path(szFileName);
		path = path.parent_path();
		path = path / "vioserial" / "vioser.inf";

		std::string cmd = "pnputil -i -a \"" + path.generic_string() + "\"";
		spdlog::info("Command to execute: " + cmd);

		std::string output = Process::exec(cmd);
		spdlog::info("Command output: " + output);
	}

	{
		fs::path path(szFileName);
		path = path.parent_path();
		path = path / "testo-guest-additions.exe";

		std::string cmd = "sc create \"Testo Guest Additions\" binPath= \"" + path.generic_string() + "\" start= auto";
		spdlog::info("Command to execute: " + cmd);

		std::string output = Process::exec(cmd);
		spdlog::info("Command output: " + output);
	}

	{
		SCManager manager;
		Service service = manager.service("Testo Guest Additions");
		service.start();
		bool started = false;
		for (size_t i = 0; i < 10; ++i) {
			SERVICE_STATUS status = service.queryStatus();
			if (status.dwCurrentState == SERVICE_RUNNING) {
				started = true;
				break;
			}
			std::this_thread::sleep_for(1s);
		}
		if (!started) {
			throw std::runtime_error("Failed to start service");
		}
	}

	spdlog::info("OK");
}

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmdline, int show) {

	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);

	fs::path path(szFileName);
	path = path.replace_extension("txt");

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.generic_string());
	auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	int argc;
	LPWSTR *szArglist = CommandLineToArgvW(GetCommandLineW(), &argc);

	std::vector<std::string> args;
	std::vector<char*> argv;
	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;
	for (size_t i = 0; i < argc; ++i) {
		args.push_back(converter.to_bytes(szArglist[i]));
		argv.push_back((char*)args.back().c_str());
	}

	try {
		using namespace clipp;

		auto cli = (
			command("install").set(selected_command, Command::Install) |
			command("help").set(selected_command, Command::ShowHelp));

		if (!parse(argc, argv.data(), cli)) {
			std::cout << make_man_page(cli, APP_NAME) << std::endl;
			return -1;
		}

		switch (selected_command) {
			case Command::Install:
				install();
				return 0;
			case Command::ShowHelp:
				std::cout << make_man_page(cli, APP_NAME) << std::endl;
				return 0;
			default:
				throw std::runtime_error("Unknown mode");
		}
	}
	catch (const ProcessError& error) {
		spdlog::error(error.what());
		spdlog::info(error.output);
		return error.exit_code;
	}
	catch (const std::exception& error) {
		spdlog::error(error.what());
		return -1;
	}
}