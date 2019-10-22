
#include <coro/Application.h>
#include <coro/CoroPool.h>
#include <coro/SignalSet.h>
#include "Interpreter.hpp"

#include "backends/dummy/DummyEnvironment.hpp"
#include "backends/vbox/VboxEnvironment.hpp"
#ifdef WIN32
#include "backends/hyperv/HypervEnvironment.hpp"
#include <wmi.hpp>
#elif __linux__
#include "backends/qemu/QemuEnvironment.hpp"
#endif

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include <clipp.h>
#include <fmt/format.h>
#include <fstream>

using namespace clipp;

struct CancelException {};


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
	Interpreter runner(file, config);
	runner.run();
}

void run_folder(const fs::path& folder, const nlohmann::json& config) {
	auto generated = generate_script(folder);
	Interpreter runner(folder, generated, config);
	runner.run();
}

int do_main(int argc, char** argv) {

#ifdef WIN32
	wmi::CoInitializer initializer;
	initializer.initalize_security();
#endif

	std::string target, prefix, test_spec, exclude, invalidate, cache_miss_policy, json_report_file;

#ifdef WIN32
	std::string hypervisor("hyperv");
#elif __linux__
	std::string hypervisor("qemu");
#elif __APPLE__
	std::string hypervisor("vsphere");
#endif

	bool stop_on_fail = false;
	bool show_help = false;

	auto cli = (
		( option("--help").set(show_help).doc("Show this help message") ) |
		(
			value("input file or folder", target),
			option("--prefix").doc("Add a prefix to all entities, thus forming a namespace") & value("prefix", prefix),
			option("--stop_on_fail").set(stop_on_fail).doc("Stop executing after first failed test"),
			option("--test_spec").doc("Run specific tests") & value("wildcard pattern", test_spec),
			option("--exclude").doc("Do not run specific tests") & value("wildcard pattern", exclude),
			option("--invalidate").doc("Invalidate specific tests") & value("wildcard pattern", invalidate),
			option("--cache_miss_policy").doc("Apply some policy when a test loses its cache (accept, skip_branch, abort)")
				& value("cache miss policy", cache_miss_policy),
			option("--json_report").doc("Generate json-formatted statistics report") & value("output file", json_report_file),
			option("--hypervisor").doc("Hypervisor type (qemu, hyperv, vsphere, vbox, dummy)") & value("hypervisor type", hypervisor)
		)
	);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		return -1;
	}

	if (show_help) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		return 0;
	}

	if (cache_miss_policy.length()) {
		if (cache_miss_policy != "accept" && cache_miss_policy != "skip_branch" && cache_miss_policy != "abort") {
			throw std::runtime_error(std::string("Unknown cache_miss_policy value: ") + cache_miss_policy);
		}
	}


	nlohmann::json config = {
		{"stop_on_fail", stop_on_fail},
		{"cache_miss_policy", cache_miss_policy},
		{"test_spec", test_spec},
		{"exclude", exclude},
		{"invalidate", invalidate},
		{"json_report_file", json_report_file},
		{"prefix", prefix}
	};

	if (!fs::exists(target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + target);
	}

	if (hypervisor == "qemu") {
#ifndef __linux__
		throw std::runtime_error("Can't use qemu hypervisor not in Linux");
#else
		env = std::make_shared<QemuEnvironment>();
#endif
	} else if (hypervisor == "vbox") {
		env = std::make_shared<VboxEnvironment>();
	} else if (hypervisor == "hyperv") {
#ifndef WIN32
		throw std::runtime_error("Can't use hyperv not in Windows");
#else
		env = std::make_shared<HyperVEnvironment>();
#endif
	} else if (hypervisor == "vsphere") {
		throw std::runtime_error("TODO");
	} else if (hypervisor == "dummy") {
		env = std::make_shared<DummyEnvironment>();
	} else {
		throw std::runtime_error(std::string("Unknown hypervisor: ") + hypervisor);
	}

	coro::CoroPool pool;
	pool.exec([&] {
		coro::SignalSet set({SIGINT, SIGTERM});
		set.wait();
		throw CancelException();
	});

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
		} catch (const CancelException&) {
			std::cout << "Cancelled" << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
