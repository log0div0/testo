
#include <coro/Application.h>
#include <coro/CoroPool.h>
#include <coro/SignalSet.h>
#include "IR/Program.hpp"
#include "Parser.hpp"

#include "backends/vbox/VboxEnvironment.hpp"
#ifdef WIN32
#include "backends/hyperv/HypervEnvironment.hpp"
#include <wmi.hpp>
#elif __linux__
#include "backends/qemu/QemuEnvironment.hpp"
#endif

#ifdef USE_CUDA
#include <license/License.hpp>
#endif

#include <iostream>
#include <thread>
#include <chrono>

#include "Utils.hpp"
#include "ModeClean.hpp"
#include <clipp.h>
#include <fstream>

using namespace clipp;

struct Interruption {};

enum class mode {run, clean, help, version};

struct console_args {
	std::string target;
	std::string prefix;
	std::string test_spec;
	std::string exclude;
	std::string invalidate;
	std::string report_folder;
	std::string license;

	std::vector<std::string> params_names;
	std::vector<std::string> params_values;

	bool stop_on_fail = false;
	bool assume_yes = false;
	bool report_logs = false;
	bool report_screenshots = false;
	bool html = false;
};

console_args args;

std::shared_ptr<Environment> env;

int run_mode() {

#ifdef USE_CUDA
	if (args.license.size()) {
		verify_license(args.license, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");
	} else {
		throw std::runtime_error("To start the program you must specify the path to the license file (--license argument)");
	}
#endif

	auto params = nlohmann::json::array();

	std::set<std::string> unique_param_names;

	for (size_t i = 0; i < args.params_names.size(); ++i) {
		auto result = unique_param_names.insert(args.params_names[i]);
		if (!result.second) {
			throw std::runtime_error("Error: param \"" + args.params_names[i] + "\" is defined multiple times as a command line argument");
		}
		nlohmann::json json_param = {
			{ "name", args.params_names[i]},
			{ "value", args.params_values[i]}
		};
		params.push_back(json_param);
	}

	nlohmann::json config = {
		{"stop_on_fail", args.stop_on_fail},
		{"assume_yes", args.assume_yes},
		{"test_spec", args.test_spec},
		{"exclude", args.exclude},
		{"invalidate", args.invalidate},
		{"report_folder", args.report_folder},
		{"report_logs", args.report_logs},
		{"report_screenshots", args.report_screenshots},
		{"html", args.html},
		{"prefix", args.prefix},
		{"params", params}
	};

	if (!fs::exists(args.target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + args.target);
	}

	auto parser = Parser::load(args.target);
	auto ast = parser.parse();
	IR::Program program(ast, config);
	program.validate();
	program.run();

	return 0;
}

int do_main(int argc, char** argv) {

#ifdef WIN32
	wmi::CoInitializer initializer;
	initializer.initalize_security();
	SetConsoleOutputCP(CP_UTF8);
#endif

#ifdef WIN32
	std::string hypervisor = "hyperv";
#elif __linux__
	std::string hypervisor = "qemu";
#elif __APPLE__
	std::string hypervisor = "vsphere";
#endif

	mode selected_mode;
	std::vector<std::string> wrong;

	auto params_defs_spec = repeatable(
		option("--param") & value("param_name", args.params_names) & value("param_value", args.params_values)
	) % "Parameters to define for test cases";

	auto run_spec = "run options" % (
		command("run").set(selected_mode, mode::run),
		value("input file or folder", args.target) % "Path to a file with testcases or to a folder with such files",
		params_defs_spec,
		(option("--prefix") & value("prefix", args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--stop_on_fail").set(args.stop_on_fail)) % "Stop executing after first failed test",
		(option("--assume_yes").set(args.assume_yes)) % "Quietly agree to run lost cache tests",
		(option("--test_spec") & value("wildcard pattern", args.test_spec)) % "Run specific tests",
		(option("--exclude") & value("wildcard pattern", args.exclude)) % "Do not run specific tests",
		(option("--invalidate") & value("wildcard pattern", args.invalidate)) % "Invalidate specific tests",
		(option("--report_folder") & value("/path/to/folder", args.report_folder)) % "Save report.json in specified folder. If folder exists it must be empty",
		(option("--report_logs").set(args.report_logs)) % "Save text output in report folder",
		(option("--report_screenshots").set(args.report_screenshots)) % "Save screenshots from failed wait actions in report folder",
		(option("--content_cksum_maxsize") & value("Size in Megabytes", content_cksum_maxsize)) % "Maximum filesize for content-based consistency checking",
		(option("--html").set(args.html)) % "Format stdout as html",
		(option("--license") & value("path", args.license)) % "Path to the license file (for GPU version only)",
		(option("--hypervisor") & value("hypervisor type", hypervisor)) % "Hypervisor type (qemu, hyperv, vbox)",
		any_other(wrong)
	);

	CleanModeArgs clean_args;

	auto clean_spec = "clean options" % (
		command("clean").set(selected_mode, mode::clean),
		(option("--prefix") & value("prefix", clean_args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--hypervisor") & value("hypervisor type", hypervisor)) % "Hypervisor type (qemu, hyperv, vbox)",
		any_other(wrong)
	);

	auto help_spec = command("help").set(selected_mode, mode::help);
	auto version_spec = command("version").set(selected_mode, mode::version);

	auto cli = (
		run_spec |
		clean_spec |
		help_spec |
		version_spec
	);

	auto res = parse(argc, argv, cli);

	if (wrong.size()) {
		for (const std::string& arg: wrong) {
			std::cerr << "Error: '" << arg << "' is not a valid argument" << std::endl;
		}
		std::cout << "Usage:" << std::endl << usage_lines(cli, argv[0]) << std::endl;
		return -1;
	}

	if (!res) {
		std::cerr << "Error: invalid command line arguments" << std::endl;
		std::cout << "Usage:" << std::endl << usage_lines(cli, argv[0]) << std::endl;
		return -1;
	}

	if (selected_mode == mode::help) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		return 0;
	}

	if (selected_mode == mode::version) {
		std::cout << "Testo framework version " << TESTO_VERSION << std::endl;
		return 0;
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
	} else {
		throw std::runtime_error(std::string("Unknown hypervisor: ") + hypervisor);
	}



	coro::CoroPool pool;
	pool.exec([&] {
		coro::SignalSet set({SIGINT, SIGTERM});
		set.wait();
		throw Interruption();
	});

	if (selected_mode == mode::clean) {
		return clean_mode(clean_args);
	} else if (selected_mode == mode::run) {
		return run_mode();
	} else {
		throw std::runtime_error("Unknown mode");
	}
}

int main(int argc, char** argv) {
	int result = 0;
	coro::Application([&]{
		try {
			result = do_main(argc, argv);
		} catch (const std::exception& error) {
			std::cerr << error << std::endl;
			result = 1;
		} catch (const Interruption&) {
			std::cerr << "Interrupted by user" << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
