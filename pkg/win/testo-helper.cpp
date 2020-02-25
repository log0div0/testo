
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

#include <shellapi.h>

namespace fs = std::filesystem;

enum class Command {
	Install,
	Uninstall,
	ShowHelp
};

#define APP_NAME "testo-guest-additions-helper"

Command selected_command;

void install() {
	spdlog::info("Install ...");

	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);
	fs::path path(szFileName);
	path = path.parent_path();

	spdlog::info("my dir is {}", path.generic_string());

	spdlog::info("OK");
}

void uninstall() {
	spdlog::info("Uninstall ...");

	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);
	fs::path path(szFileName);
	path = path.parent_path();

	spdlog::info("my dir is {}", path.generic_string());

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
			command("uninstall").set(selected_command, Command::Uninstall) |
			command("help").set(selected_command, Command::ShowHelp));

		if (!parse(argc, argv.data(), cli)) {
			std::cout << make_man_page(cli, APP_NAME) << std::endl;
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
	catch (const std::exception& error) {
		spdlog::error(error.what());
		return -1;
	}
}
