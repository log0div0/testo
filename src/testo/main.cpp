
#include <coro/Application.h>
#include <coro/CoroPool.h>
#include <coro/SignalSet.h>

#include "backends/vbox/VboxEnvironment.hpp"
#ifdef WIN32
#include "backends/hyperv/HypervEnvironment.hpp"
#include <wmi.hpp>
#elif __linux__
#include "backends/qemu/QemuEnvironment.hpp"
#endif

#include <iostream>

#include "ModeClean.hpp"
#include "ModeRun.hpp"
#include <clipp.h>

using namespace clipp;

struct Interruption {};

enum class mode {run, clean, help, version};

std::shared_ptr<Environment> env;

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

	RunModeArgs run_args;

	auto params_defs_spec = repeatable(
		option("--param") & value("param_name", run_args.params_names) & value("param_value", run_args.params_values)
	) % "Parameters to define for test cases";

	auto run_spec = "run options" % (
		command("run").set(selected_mode, mode::run),
		value("input file or folder", run_args.target) % "Path to a file with testcases or to a folder with such files",
		params_defs_spec,
		(option("--prefix") & value("prefix", run_args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--stop_on_fail").set(run_args.stop_on_fail)) % "Stop executing after first failed test",
		(option("--assume_yes").set(run_args.assume_yes)) % "Quietly agree to run lost cache tests",
		(option("--test_spec") & value("wildcard pattern", run_args.test_spec)) % "Run specific tests",
		(option("--exclude") & value("wildcard pattern", run_args.exclude)) % "Do not run specific tests",
		(option("--invalidate") & value("wildcard pattern", run_args.invalidate)) % "Invalidate specific tests",
		(option("--report_folder") & value("/path/to/folder", run_args.report_folder)) % "Save report.json in specified folder. If folder exists it must be empty",
		(option("--report_logs").set(run_args.report_logs)) % "Save text output in report folder",
		(option("--report_screenshots").set(run_args.report_screenshots)) % "Save screenshots from failed wait actions in report folder",
		(option("--content_cksum_maxsize") & value("Size in Megabytes", content_cksum_maxsize)) % "Maximum filesize for content-based consistency checking",
		(option("--html").set(run_args.html)) % "Format stdout as html",
		(option("--license") & value("path", run_args.license)) % "Path to the license file (for GPU version only)",
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
		return run_mode(run_args);
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
