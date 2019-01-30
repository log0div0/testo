
#include "Interpreter.hpp"
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"

static void print_usage() {
	std::cout << "Usage: \n";
	std::cout << "testo <input file>\n";
}

int main(int argc, char** argv) {
	try {
		if (argc != 2) {
			print_usage();
		}

		fs::path src_file(argv[1]);

		if (!fs::exists(src_file)) {
			throw std::runtime_error("Fatal error: file doesn't exist: " + src_file.generic_string());
		}

		VboxEnvironment env;
		Interpreter runner(env, src_file);
		runner.run();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}



	return 0;
}
