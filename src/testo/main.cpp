
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

#include <license/License.hpp>

using namespace clipp;

struct Interruption {};

enum mode {run, clean};

struct console_args {
	mode selected_mode;
	std::string target;
	std::string prefix;
	std::string test_spec;
	std::string exclude;
	std::string invalidate;
	std::string hypervisor;
	std::string report_folder;
	std::string license;

	std::vector<std::string> params_names;
	std::vector<std::string> params_values;

	bool show_help = false;
	bool stop_on_fail = false;
	bool assume_yes = false;
	bool report_logs = false;
	bool report_screenshots = false;
};

console_args args;

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

int clean_mode() {
	//cleanup networks
	for (auto& network_folder: fs::directory_iterator(env->network_metadata_dir())) {
		for (auto& file: fs::directory_iterator(network_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(network_folder).filename()) {
					auto config = nlohmann::json::parse(get_metadata(file, "network_config"));

					auto network_controller = env->create_network_controller(config);
					if (network_controller->prefix() == args.prefix) {
						network_controller->undefine();
						std::cout << "Deleted network " << network_controller->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cout << "Couldn't remove network " << fs::path(file).filename() << std::endl;
				std::cout << error << std::endl;
			}

		}
	}

	//cleanup flash drives
	for (auto& flash_drive_folder: fs::directory_iterator(env->flash_drives_metadata_dir())) {
		for (auto& file: fs::directory_iterator(flash_drive_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(flash_drive_folder).filename()) {
					auto config = nlohmann::json::parse(get_metadata(file, "fd_config"));
					auto flash_drive_contoller = env->create_flash_drive_controller(config);
					if (flash_drive_contoller->prefix() == args.prefix) {
						flash_drive_contoller->undefine();
						std::cout << "Deleted flash drive " << flash_drive_contoller->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cout << "Couldn't remove flash drive " << fs::path(file).filename() << std::endl;
				std::cout << error << std::endl;
			}
		}
	}

	//cleanup virtual machines
	for (auto& vm_folder: fs::directory_iterator(env->vm_metadata_dir())) {
		for (auto& file: fs::directory_iterator(vm_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(vm_folder).filename()) {
					auto config = nlohmann::json::parse(get_metadata(file, "vm_config"));
					auto vm_contoller = env->create_vm_controller(config);
					if (vm_contoller->prefix() == args.prefix) {
						vm_contoller->undefine();
						std::cout << "Deleted virtual machine " << vm_contoller->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cout << "Couldn't remove virtual machine " << fs::path(file).filename() << std::endl;
				std::cout << error << std::endl;
			}

		}
	}
	return 0;
}

int run_mode() {
	if (!args.license.size()) {
		std::cout << "Необходимо указать путь к файлу с лицензией (параметр --license)" << std::endl;
		return 1;
	}

	verify_license(args.license, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");

	auto params = nlohmann::json::array();

	for (size_t i = 0; i < args.params_names.size(); ++i) {
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
		{"prefix", args.prefix},
		{"params", params}
	};

	if (!fs::exists(args.target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + args.target);
	}

	if (fs::is_regular_file(args.target)) {
		run_file(args.target, config);
	} else if (fs::is_directory(args.target)) {
		run_folder(args.target, config);
	} else {
		throw std::runtime_error(std::string("Fatal error: unknown target type: ") + args.target);
	}

	return 0;
}

int do_main(int argc, char** argv) {

#ifdef WIN32
	wmi::CoInitializer initializer;
	initializer.initalize_security();
#endif
#ifdef WIN32
	args.hypervisor = "hyperv";
#elif __linux__
	args.hypervisor = "qemu";
#elif __APPLE__
	args.hypervisor = "vsphere";
#endif

	auto params_defs_spec = repeatable(
		option("--param") & value("param_name", args.params_names) & value("param_value", args.params_values)
	) % "Parameters to define for test cases";

	auto run_spec = "run options" % (
		command("run").set(args.selected_mode, mode::run),
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
		(option("--license") & value("path", args.license)) % "Path to license file",
		(option("--hypervisor") & value("hypervisor type", args.hypervisor)) % "Hypervisor type (qemu, hyperv, vsphere, vbox, dummy)"
	);

	auto clean_spec = "clean options" % (
		command("clean").set(args.selected_mode, mode::clean),
		(option("--prefix") & value("prefix", args.prefix)) % "Add a prefix to all entities, thus forming a namespace",
		(option("--hypervisor") & value("hypervisor type", args.hypervisor)) % "Hypervisor type (qemu, hyperv, vsphere, vbox, dummy)"
	);

	auto cli = ( run_spec | clean_spec | command("help").set(args.show_help) );

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		return -1;
	}

	if (args.show_help) {
		std::cout << make_man_page(cli, "testo") << std::endl;
		return 0;
	}

	if (args.hypervisor == "qemu") {
#ifndef __linux__
		throw std::runtime_error("Can't use qemu hypervisor not in Linux");
#else
		env = std::make_shared<QemuEnvironment>();
#endif
	} else if (args.hypervisor == "vbox") {
		env = std::make_shared<VboxEnvironment>();
	} else if (args.hypervisor == "hyperv") {
#ifndef WIN32
		throw std::runtime_error("Can't use hyperv not in Windows");
#else
		env = std::make_shared<HyperVEnvironment>();
#endif
	} else if (args.hypervisor == "vsphere") {
		throw std::runtime_error("TODO");
	} else if (args.hypervisor == "dummy") {
		env = std::make_shared<DummyEnvironment>();
	} else {
		throw std::runtime_error(std::string("Unknown hypervisor: ") + args.hypervisor);
	}

	coro::CoroPool pool;
	pool.exec([&] {
		coro::SignalSet set({SIGINT, SIGTERM});
		set.wait();
		throw Interruption();
	});

	if (args.selected_mode == mode::clean) {
		return clean_mode();
	} else if (args.selected_mode == mode::run) {
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
			std::cout << error << std::endl;
			result = 1;
		} catch (const Interruption&) {
			std::cout << "Interrupted by user" << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
