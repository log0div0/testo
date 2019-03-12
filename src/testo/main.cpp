
#include "Interpreter.hpp"
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include <coro/Application.h>

static void print_usage() {
	std::cout << "Usage: \n";
	std::cout << "testo <input file>\n";
}

int do_main(int argc, char** argv) {
	if (argc != 2) {
		print_usage();
	}

	fs::path src_file(argv[1]);

	if (!fs::exists(src_file)) {
		throw std::runtime_error("Fatal error: file doesn't exist: " + src_file.generic_string());
	}

	QemuEnvironment env;
	Interpreter runner(env, src_file);
	runner.run();
	return 0;
}

int main(int argc, char** argv) {
	int result = 0;
	coro::Application([&]{
		try {
			result = do_main(argc, argv);
		} catch (const std::exception& error) {
			std::cout << error.what() << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
