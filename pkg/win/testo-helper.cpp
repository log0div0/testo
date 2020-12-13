
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

using namespace std::chrono_literals;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <clipp.h>

#include <tchar.h>
#include <shellapi.h>

#include "winapi/Functions.hpp"
#include "winapi/RegKey.hpp"
#include "winapi/UTF.hpp"

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

enum class Command {
	Install,
	Uninstall,
	ShowHelp
};

#define APP_NAME "testo-helper"

Command selected_command;

std::string usersid;

std::vector<std::string> split(std::string strToSplit) {
	std::stringstream ss(strToSplit);
	std::string item;
	std::vector<std::string> splittedStrings;
	while (std::getline(ss, item, ';'))
	{
		if (item.size()) {
			splittedStrings.push_back(item);
		}
	}
	return splittedStrings;
}

std::string join(const std::vector<std::string>& strsToJoin) {
	std::string result;
	for (size_t i = 0; i < strsToJoin.size(); ++i) {
		result += strsToJoin[i];
		result += ';';
	}
	return result;
}

void install() {
	spdlog::info("Install ...");

	{
		fs::path testo_dir = fs::path(winapi::get_module_file_name()).parent_path();
		spdlog::info("testo_dir is {}", testo_dir.string());

		winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment");
		// winapi::RegKey regkey(HKEY_USERS, usersid + "\\Environment");
		std::string env_path = regkey.query_str("PATH");
		spdlog::info("current PATH is {}", env_path);

		std::vector<std::string> splitted_path = split(env_path);
		auto it = std::find(splitted_path.begin(), splitted_path.end(), testo_dir.string());
		if (it == splitted_path.end()) {
			splitted_path.push_back(testo_dir.string());

			env_path = join(splitted_path);
			spdlog::info("new PATH is {}", env_path);

			regkey.set_expand_str("PATH", env_path);
		}
	}

	SendMessage(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)_T("Environment"));

	spdlog::info("OK");
}

void uninstall() {
	spdlog::info("Uninstall ...");

	{
		fs::path testo_dir = fs::path(winapi::get_module_file_name()).parent_path();
		spdlog::info("testo_dir is {}", testo_dir.string());

		winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment");
		// winapi::RegKey regkey(HKEY_USERS, usersid + "\\Environment");
		std::string env_path = regkey.query_str("PATH");
		spdlog::info("current PATH is {}", env_path);

		std::vector<std::string> splitted_path = split(env_path);
		auto it = std::find(splitted_path.begin(), splitted_path.end(), testo_dir.string());
		if (it != splitted_path.end()) {
			splitted_path.erase(it);

			env_path = join(splitted_path);
			spdlog::info("new PATH is {}", env_path);

			regkey.set_expand_str("PATH", env_path);
		}
	}

	SendMessage(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)_T("Environment"));

	spdlog::info("OK");
}

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmdline, int show) {

	fs::path log_path = fs::path(winapi::get_module_file_name()).replace_extension("txt");
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

		auto install_spec = (
			command("install").set(selected_command, Command::Install),
			(option("--usersid") & value("sid", usersid))
		);

		auto uninstall_spec = (
			command("uninstall").set(selected_command, Command::Uninstall),
			(option("--usersid") & value("sid", usersid))
		);

		auto cli = (
			install_spec |
			uninstall_spec |
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
	catch (const std::exception& error) {
		spdlog::error(error.what());
		return -1;
	}
}
