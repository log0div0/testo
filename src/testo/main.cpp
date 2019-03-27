
#include "Interpreter.hpp"
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include <coro/Application.h>
#include <clipp.h>

using namespace clipp;

int do_main(int argc, char** argv) {
	std::string src_file;
	bool stop_on_fail = false;

	auto cli = (
		value("input file", src_file),
		option("--stop_on_fail").set(stop_on_fail).doc("Stop executing after first failed test")
	);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		throw std::runtime_error("");
	}

	if (!fs::exists(src_file)) {
		throw std::runtime_error("Fatal error: file doesn't exist: " + src_file);
	}

	nlohmann::json config = {
		{"stop_on_fail", stop_on_fail}
	};

	QemuEnvironment env;
	Interpreter runner(env, src_file, config);
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
