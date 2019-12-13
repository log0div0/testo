#include <iostream>
#include <nlohmann/json.hpp>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "Server.hpp"

int main(int argc, char** argv) {
	Server commands_handler("/dev/ttyS1");
	commands_handler.run();
	return 0;
}