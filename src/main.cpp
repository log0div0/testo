
#include <Interpreter.hpp>
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include <Utils.hpp>

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
			throw std::runtime_error(std::string("Fatal error: file doesn't exist: ") + std::string(src_file));
		}

		if (src_file.is_relative()) {
			src_file = fs::canonical(src_file);
		}
		
		Interpreter runner(src_file);
		runner.run();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}

	

	return 0;
}
