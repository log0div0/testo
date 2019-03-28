
#include "Interpreter.hpp"
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include <coro/Application.h>
#include <clipp.h>
#include <fmt/format.h>
#include <fstream>

using namespace clipp;

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
	QemuEnvironment env;
	Interpreter runner(env, file, config);
	runner.run();
}

void run_folder(const fs::path& folder, const nlohmann::json& config) {
	auto generated = generate_script(folder);
	fs::path generated_path = folder / "testo-generated";
	std::ofstream output_stream(generated_path, std::ofstream::out);
	if (!output_stream) {
		throw std::runtime_error(std::string("Can't create file for writing generated script: ") + generated_path.generic_string());
	}
	output_stream << generated;
	output_stream.close();

	run_file(generated_path, config);
}

int do_main(int argc, char** argv) {
	std::string target, test_spec("");
	bool stop_on_fail = false;

	auto cli = (
		value("input file", target),
		option("--stop_on_fail").set(stop_on_fail).doc("Stop executing after first failed test"),
		option("--test_spec").doc("Run specific test") & value("test name", test_spec)
	);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		throw std::runtime_error("");
	}

	nlohmann::json config = {
		{"stop_on_fail", stop_on_fail},
		{"test_spec", test_spec}
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
			std::cout << error.what() << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
