
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <clipp.h>

#include "process/Process.hpp"

namespace fs = std::filesystem;

enum class Command {
	Install,
	ShowHelp
};

#define APP_NAME "testo-guest-additions-helper"
#define LOG_FILE_PATH "C:/log.txt"

std::string targetdir;
Command selected_command;

void install() {
	spdlog::info("Install ...");

	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);

	fs::path path(szFileName);
	path = path.parent_path();
	path = path / "vioserial" / "vioser.inf";

	std::string cmd = "pnputil -i -a \"" + path.generic_string() + "\"";
	spdlog::info("Command to execute: " + cmd);

	std::string output = Process::exec(cmd);
	spdlog::info("Command output: " + output);

	spdlog::info("OK");
}

int main(int argc, char** argv) {

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(LOG_FILE_PATH);
	auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	try {
		using namespace clipp;

		auto cli = (
			command("install").set(selected_command, Command::Install) |
			command("help").set(selected_command, Command::ShowHelp));

		if (!parse(argc, argv, cli)) {
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
