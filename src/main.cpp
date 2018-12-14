
#include <Interpreter.hpp>
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

static void print_usage() {
	std::cout << "Usage: \n";
	std::cout << "testo <input file>\n";
}

int main(int argc, char** argv) {
	try {
		if (argc != 2) {
			print_usage();
		}
		Interpreter runner(argv[1]);
		runner.run();		
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}

	

	return 0;
}
