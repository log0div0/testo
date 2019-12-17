#include <iostream>
#include <string.h>
#include "spdlog/spdlog.h"
#include "spdlog/logger.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "Server.hpp"

int main(int argc, char** argv) {
	auto logger = spdlog::basic_logger_mt("basic_logger", "/var/log/testo-guest-additions.log");
	spdlog::set_default_logger(logger);
	Server commands_handler("/dev/virtio-ports/negotiator.0");
	commands_handler.run();
	return 0;
}