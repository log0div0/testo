
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

namespace fs = std::filesystem;

enum class Command {
	Install,
	ShowHelp
};

#define APP_NAME "testo-guest-additions-helper"

Command selected_command;

void install() {
	spdlog::info("Install ...");

	{
		fs::path path = winapi::get_module_file_name();
		path = path.parent_path();
		path = path / "vioserial" / "vioser.inf";

		std::string cmd = "pnputil -i -a \"" + path.generic_string() + "\"";
		spdlog::info("Command to execute: " + cmd);

		std::string output = Process::exec(cmd);
		spdlog::info("Command output: " + output);
	}

	{
		fs::path path = winapi::get_module_file_name();
		path = path.parent_path();
		path = path / "testo-guest-additions.exe";

		std::string cmd = "sc create \"Testo Guest Additions\" binPath= \"" + path.generic_string() + "\" start= auto";
		spdlog::info("Command to execute: " + cmd);

		std::string output = Process::exec(cmd);
		spdlog::info("Command output: " + output);
	}

	{
		winapi::SCManager manager;
		winapi::Service service = manager.service("Testo Guest Additions");
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
			command("help").set(selected_command, Command::ShowHelp));

		if (!parse(argc, argv.data(), cli)) {
			spdlog::error("failed to parse args");
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
