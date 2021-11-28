
#include <coro/Application.h>
#include <coro/CoroPool.h>
#include <coro/SignalSet.h>
#include <coro/Finally.h>

#ifdef WIN32
#include "backends/hyperv/HypervEnvironment.hpp"
#include <wmi/CoInitializer.hpp>
#include <winapi/Functions.hpp>
#elif __linux__
#include "backends/qemu/QemuEnvironment.hpp"
#elif __APPLE__
#include "backends/Environment.hpp"
#endif

#include <iostream>

#include "ModeClean.hpp"
#include "ModeRun.hpp"

#include "Exceptions.hpp"
#include "Logger.hpp"

#include <clipp.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

using namespace clipp;

struct Interruption {};

enum class mode {
	run,
	clean,
	help,
	version
};

void init_logs() {
#ifdef __linux__
	std::string log_file_path = "/var/log/testo.log";
#else
	fs::path parent_folder = fs::path(winapi::get_module_file_name()).parent_path();
	fs::path logs_path = parent_folder / "testo.txt";
	std::string log_file_path = logs_path.generic_string();
#endif

	auto max_size = 1048576 * 10;
    auto max_files = 3;
	auto logger = spdlog::rotating_logger_mt("basic_logger", log_file_path, max_size, max_files);
	logger->set_level(spdlog::level::trace);
	logger->flush_on(spdlog::level::trace);
	spdlog::set_default_logger(logger);
	spdlog::info("logger is initialized");
}

void check_privileges() {
	TRACE();
#ifdef __linux__
	uid_t uid = geteuid();
	if (uid != 0) {
		throw std::runtime_error("Please run me as root");
	}
#endif
}

std::shared_ptr<Environment> env;

void init_env(const std::string& hypervisor) {
	TRACE();
	if (hypervisor == "qemu") {
#ifndef __linux__
		throw std::runtime_error("Qemu is only supported on Linux");
#else
		env = std::make_shared<QemuEnvironment>();
#endif
	} else if (hypervisor == "hyperv") {
#ifndef WIN32
		throw std::runtime_error("HyperV is only supported on Windows");
#else
		env = std::make_shared<HyperVEnvironment>();
#endif
	} else {
		throw std::runtime_error("Unknown hypervisor: " + hypervisor);
	}
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

	RunModeArgs run_args;

	auto test_spec_filer = [&](const std::string& arg) {
		run_args.test_name_filters.push_back({TestNameFilter::Type::test_spec, arg});
		return true;
	};
	auto exclude_filer = [&](const std::string& arg) {
		run_args.test_name_filters.push_back({TestNameFilter::Type::exclude, arg});
		return true;
	};

	auto params_defs_spec = repeatable(
		option("--param") & value("param_name", run_args.params_names) & value("param_value", run_args.params_values)
	) % "Parameters to define for test cases";

	auto test_spec = repeatable(
		option("--test_spec") & value(test_spec_filer, "wildcard pattern")
	) % "Run specific tests";

	auto exclude_spec = repeatable(
		option("--exclude") & value(exclude_filer, "wildcard pattern")
	) % "Do not run specific tests";

	auto run_spec = "run options" % (
		command("run").set(selected_mode, mode::run),
		value("input file or folder", run_args.target) % "Path to a file with testcases or to a folder with such files",
		params_defs_spec,
		test_spec,
		exclude_spec,
		(option("--prefix") & value("prefix", run_args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--stop_on_fail").set(run_args.stop_on_fail)) % "Stop executing after first failed test",
		(option("--assume_yes").set(run_args.assume_yes)) % "Quietly agree to run lost cache tests",
		(option("--invalidate") & value("wildcard pattern", run_args.invalidate)) % "Invalidate specific tests",
		(option("--report_folder") & value("/path/to/folder", run_args.report_folder)) % "Save report.json in specified folder",
		(option("--report_format") & value("format id", run_args.report_format)) % "The format of the report to be used (native, allure)",
		(option("--content_cksum_maxsize") & value("Size in Megabytes", content_cksum_maxsize)) % "Maximum filesize for content-based consistency checking",
		(option("--html").set(run_args.html)) % "Format stdout as html",
		(option("--nn_server") & value("ip:port", run_args.nn_server_endpoint)) % "ip:port of the nn_server (defualt is 127.0.0.1:8156)",
		(option("--hypervisor") & value("hypervisor type", hypervisor)) % "Hypervisor type (qemu, hyperv)",
		(option("--dry").set(run_args.dry)) % "Do only semantic checks, do not actually run any tests",
		any_other(wrong)
	);

	CleanModeArgs clean_args;

	auto clean_spec = "clean options" % (
		command("clean").set(selected_mode, mode::clean),
		(option("--prefix") & value("prefix", clean_args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--assume_yes").set(clean_args.assume_yes)) % "Quietly agree to erase all the virtual entities",
		(option("--hypervisor") & value("hypervisor type", hypervisor)) % "Hypervisor type (qemu, hyperv)",
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
		std::cout << "Usage:" << std::endl << usage_lines(cli, "testo") << std::endl;
		return -1;
	}

	if (!res) {
		std::cerr << "Error: invalid command line arguments" << std::endl;
		std::cout << "Usage:" << std::endl << usage_lines(cli, "testo") << std::endl;
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

	init_logs();
	TRACE();
	check_privileges();
	init_env(hypervisor);
	coro::Finally cleanup([&] {
		env.reset();
	});

	coro::CoroPool pool;
	pool.exec([&] {
		coro::SignalSet set({SIGINT, SIGTERM});
		set.wait();
		throw Interruption();
	});

	if (selected_mode == mode::clean) {
		return clean_mode(clean_args);
	} else if (selected_mode == mode::run) {
		run_args.params_names.push_back("TESTO_HYPERVISOR");
		run_args.params_values.push_back(hypervisor);
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
		} catch (const TestFailedException& error) {
			std::cout << error << std::endl;
			result = 1;
		}  catch (const std::exception& error) {
			std::cerr << error << std::endl;
			result = 2;
		} catch (const Interruption&) {
			std::cerr << "Interrupted by user" << std::endl;
			result = 3;
		}
	}).run();

	return result;
}
