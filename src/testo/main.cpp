
#include <coro/Application.h>
#include "Interpreter.hpp"

#ifdef WIN32
#include "backends/hyperv/HypervEnvironment.hpp"
#include <wmi.hpp>
#elif __linux__
#include "backends/qemu/QemuEnvironment.hpp"
#elif __APPLE__
#include "backends/vbox/VboxEnvironment.hpp"
#endif

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include <clipp.h>
#include <fmt/format.h>
#include <fstream>

using namespace clipp;

std::shared_ptr<Environment> env;

std::string generate_script(const fs::path& folder, const fs::path& current_prefix = ".") {
	std::string result("");
	for (auto& file: fs::directory_iterator(folder)) {
		if (fs::is_regular_file(file)) {
			if (fs::path(file).extension() == ".testo") {
				result += fmt::format("include \"{}\"\n", fs::path(current_prefix / fs::path(file).filename()).generic_string());
			}
		} else if (fs::is_directory(file)) {
			result += generate_script(file, current_prefix / fs::path(file).filename());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}

	return result;
}

void run_file(const fs::path& file, const nlohmann::json& config) {
#ifdef WIN32
	env = std::make_shared<HyperVEnvironment>();
#elif __linux__
	env = std::make_shared<QemuEnvironment>();
#elif __APPLE__
	env = std::make_shared<VboxEnvironment>();
#endif
	Interpreter runner(file, config);
	runner.run();
}

void run_folder(const fs::path& folder, const nlohmann::json& config) {
	auto generated = generate_script(folder);

#ifdef WIN32
	env = std::make_shared<HyperVEnvironment>();
#elif __linux__
	env = std::make_shared<QemuEnvironment>();
#elif __APPLE__
	env = std::make_shared<VboxEnvironment>();
#endif
	Interpreter runner(folder, generated, config);
	runner.run();
}

int do_main(int argc, char** argv) {

#ifdef WIN32
	wmi::CoInitializer initializer;
	initializer.initalize_security();
#endif

	std::string target, test_spec, exclude, invalidate;
	bool stop_on_fail = false;

	auto cli = (
		value("input file", target),
		option("--stop_on_fail").set(stop_on_fail).doc("Stop executing after first failed test"),
		option("--test_spec").doc("Run specific tests") & value("wildcard pattern", test_spec),
		option("--exclude").doc("Do not run specific tests") & value("wildcard pattern", exclude),
		option("--invalidate").doc("invalidate specific tests") & value("wildcard pattern", invalidate)
	);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		throw std::runtime_error("");
	}

	nlohmann::json config = {
		{"stop_on_fail", stop_on_fail},
		{"test_spec", test_spec},
		{"exclude", exclude},
		{"invalidate", invalidate}
	};

	if (!fs::exists(target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + target);
	}

	if (fs::is_regular_file(target)) {
		run_file(target, config);
	} else if (fs::is_directory(target)) {
		run_folder(target, config);
	} else {
		throw std::runtime_error(std::string("Fatal error: unknown target type: ") + target);
	}

	return 0;
}

int main(int argc, char** argv) {
	int result = 0;
	coro::Application([&]{
		try {
			result = do_main(argc, argv);
		} catch (const std::exception& error) {
			std::cout << error << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
