
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

using namespace std::chrono_literals;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <clipp.h>

#include "process/Process.hpp"
#include "winapi.hpp"

#include <shellapi.h>
#include <Lm.h>

namespace fs = std::filesystem;

enum class Command {
	Install,
	Uninstall,
	ShowHelp
};

#define APP_NAME "testo-guest-additions-helper"

Command selected_command;

WKSTA_INFO_100 get_os_info() {
	LPBYTE pinfoRawData = nullptr;
	if (NERR_Success != NetWkstaGetInfo(NULL, 100, &pinfoRawData)) {
		throw std::runtime_error("NetWkstaGetInfo failed");
	}
	WKSTA_INFO_100 info = *(WKSTA_INFO_100*)pinfoRawData;
	NetApiBufferFree(pinfoRawData);
	return info;
}

std::string get_os() {
	WKSTA_INFO_100 info = get_os_info();
	spdlog::info("OS VERSION: major {} minor {}", info.wki100_ver_major, info.wki100_ver_minor);
	if (info.wki100_ver_major == 10) {
		return "w10";
	} else if ((info.wki100_ver_major == 6) && (info.wki100_ver_minor == 3)) {
		return "w8.1";
	} else if ((info.wki100_ver_major == 6) && (info.wki100_ver_minor == 2)) {
		return "w8";
	} else if ((info.wki100_ver_major == 6) && (info.wki100_ver_minor == 1)) {
		return "w7";
	} else {
		throw std::runtime_error("Unsupported os");
	}
}

void install_driver() {
	fs::path path = winapi::get_module_file_name();
	path = path.parent_path();
	path = path / "vioserial" / get_os() / "vioser.inf";

	std::string cmd = "pnputil -i -a \"" + path.generic_string() + "\"";
	spdlog::info("Command to execute: " + cmd);

	std::string output = Process::exec(cmd);
	spdlog::info("Command output: " + output);
}

void create_service() {
	fs::path path = winapi::get_module_file_name();
	path = path.parent_path();
	path = path / "testo-guest-additions.exe";

	std::string cmd = "sc create \"Testo Guest Additions\" binPath= \"" + path.generic_string() + "\" start= auto";
	spdlog::info("Command to execute: " + cmd);

	std::string output = Process::exec(cmd);
	spdlog::info("Command output: " + output);
}

void start_service() {
	winapi::SCManager manager;
	winapi::Service service = manager.service("Testo Guest Additions");
	service.start();
	for (size_t i = 0; i < 10; ++i) {
		SERVICE_STATUS status = service.queryStatus();
		if (status.dwCurrentState == SERVICE_RUNNING) {
			return;
		}
		std::this_thread::sleep_for(1s);
	}
	throw std::runtime_error("Failed to start service");
}

void install() {
	spdlog::info("Install ...");

	install_driver();
	create_service();

	spdlog::info("Sleeping 10s");
	std::this_thread::sleep_for(10s);
	spdlog::info("Sleeping done");

	start_service();

	spdlog::info("OK");
}

void stop_service() {
	winapi::SCManager manager;
	winapi::Service service = manager.service("Testo Guest Additions");

	SERVICE_STATUS status = service.queryStatus();
	if (status.dwCurrentState == SERVICE_STOPPED) {
		spdlog::info("Service is already stopped");
		return;
	} else if (status.dwCurrentState == SERVICE_STOP_PENDING) {
		spdlog::info("Service stop pending ...");
	} else {
		spdlog::info("Sending stop signal to service ...");
		service.control(SERVICE_CONTROL_STOP);
	}

	for (size_t i = 0; i < 10; ++i) {
		SERVICE_STATUS status = service.queryStatus();
		if (status.dwCurrentState == SERVICE_STOPPED) {
			return;
		}
	}

	throw std::runtime_error("Failed to stop service");
}

void delete_service() {
	std::string cmd = "sc delete \"Testo Guest Additions\"";
	spdlog::info("Command to execute: " + cmd);

	std::string output = Process::exec(cmd);
	spdlog::info("Command output: " + output);
}

void uninstall_driver() {
	fs::path path = winapi::get_module_file_name();
	path = path.parent_path();
	path = path / "vioserial" / get_os() / "vioser.inf";

	std::string cmd = "pnputil -d \"" + path.generic_string() + "\"";
	spdlog::info("Command to execute: " + cmd);

	std::string output = Process::exec(cmd);
	spdlog::info("Command output: " + output);
}

void uninstall() {
	spdlog::info("Uninstall ...");

	stop_service();
	delete_service();
	uninstall_driver();
}

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmdline, int show) {

	fs::path log_path = winapi::get_module_file_name().replace_extension("txt");
	auto logger = spdlog::basic_logger_mt("basic_logger", log_path.string());
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	int argc;
	LPWSTR *szArglist = CommandLineToArgvW(GetCommandLineW(), &argc);

	std::vector<std::string> args;
	std::vector<char*> argv;
	spdlog::info("argc = {}", argc);
	for (size_t i = 0; i < argc; ++i) {
		args.push_back(winapi::utf16_to_utf8(szArglist[i]));
		spdlog::info("arg {}: {}", i, args.back());
	}
	for (auto& arg: args) {
		argv.push_back((char*)arg.c_str());
	}

	try {
		using namespace clipp;

		auto cli = (
			command("install").set(selected_command, Command::Install) |
			command("uninstall").set(selected_command, Command::Uninstall) |
			command("help").set(selected_command, Command::ShowHelp));

		if (!parse(argc, argv.data(), cli)) {
			spdlog::error("failed to parse args");
			return -1;
		}

		switch (selected_command) {
			case Command::Install:
				install();
				return 0;
			case Command::Uninstall:
				uninstall();
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
