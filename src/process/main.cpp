
#include <iostream>
#include "Process.hpp"

int main(int argc, char** argv) {
	try {
		std::cout << Process::exec("ping -c 2 ya.ru") << std::endl;
	} catch (const ProcessError& error) {
		std::cerr << error.exit_code << std::endl;
		std::cerr << error.output << std::endl;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
	return 0;
}
