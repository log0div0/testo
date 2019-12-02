
#include "Reporter.hpp"
#include "backends/Environment.hpp"
#include <rang.hpp>
#include <fstream>
#include <fmt/format.h>

template <typename Duration>
std::string duration_to_str(Duration duration) {

	auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
	duration -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
	duration -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
	auto result = fmt::format("{}h:{}m:{}s", h.count(), m.count(), s.count());

	return result;
}

Reporter::Reporter(const nlohmann::json& config) {
	prefix = config.at("prefix").get<std::string>();
}

void Reporter::init(
		const std::list<std::shared_ptr<AST::Test>>& _tests_to_run,
		const std::vector<std::shared_ptr<AST::Test>>& _up_to_date_tests,
		const std::vector<std::shared_ptr<AST::Test>>& _ignored_tests)
{
	start_timestamp = std::chrono::system_clock::now();

	time_t tt = std::chrono::system_clock::to_time_t(start_timestamp);
	tm local_tm = *localtime(&tt);
	std::stringstream time_spec;
	time_spec << prefix;
	time_spec << local_tm.tm_year + 1900 << '-';
	time_spec << local_tm.tm_mon + 1 << '-';
	time_spec << local_tm.tm_mday << '_';
	time_spec << local_tm.tm_hour << ':';
	time_spec << local_tm.tm_min << ':';
	time_spec << local_tm.tm_sec;

	report_folder = env->reports_dir() / time_spec.str();

	fs::create_directories(report_folder);

	summary_output_file = std::ofstream(report_folder / "summary.txt");

	for (auto test: _tests_to_run) {
		tests_to_run.push_back(std::shared_ptr<Test>(new Test(test, report_folder)));
	}
	for (auto test: _up_to_date_tests) {
		up_to_date_tests.push_back(std::shared_ptr<Test>(new Test(test)));
	}
	for (auto test: _ignored_tests) {
		ignored_tests.push_back(std::shared_ptr<Test>(new Test(test)));
	}

	auto tests_num = tests_to_run.size() + up_to_date_tests.size();
	if (tests_num != 0) {
		progress_step = (float)100 / tests_num;
	} else {
		progress_step = 100;
	}

	if (up_to_date_tests.size()) {
		report("UP-TO-DATE TESTS:\n", blue, true);
		for (auto test: up_to_date_tests) {
			current_progress += progress_step;
			report(fmt::format("{}\n", test->name), magenta);
		}
	}

	if (tests_to_run.size()) {
		std::cout << rang::fgB::blue << rang::style::bold;
		report("TESTS TO RUN:\n", style::blue, true);
		for (auto test: tests_to_run) {
			report(fmt::format("{}\n", test->name), magenta);
		}
	}
}

void Reporter::finish() {
	finish_timestamp = std::chrono::system_clock::now();

	print_statistics();
	auto json_report_file = report_folder / "report.json";

	auto path = fs::absolute(json_report_file);
	auto report = create_json_report();
	fs::create_directories(path.parent_path());
	std::ofstream file(path);
	file << report.dump(2);
}

void Reporter::prepare_environment() {
	current_test = tests_to_run.front();
	tests_to_run.pop_front();

	report(fmt::format("{} Preparing the environment for test ", progress()), blue);
	report(fmt::format("{}\n", current_test->name), yellow);

	current_test->start_timestamp = std::chrono::system_clock::now();
}

void Reporter::run_test() {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Running test "
		<< rang::fg::yellow << current_test->name
		<< rang::style::reset << std::endl;
}

void Reporter::skip_failed_test(const std::string& failed_parent) {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();

	report(fmt::format("{} Skipping test ", progress()), red);
	report(current_test->name, yellow);
	report(" because his parent ", red);
	report(failed_parent, yellow);
	report(" failed\n", red);

	failed_tests.push_back(current_test);
	current_test = nullptr;
}

void Reporter::test_passed() {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();
	auto duration = duration_to_str(current_test->stop_timestamp - current_test->start_timestamp);

	report(fmt::format("{} Test ", progress()), green, true);
	report(current_test->name, yellow, true);
	report(fmt::format(" PASSED in {}\n", duration), green, true);

	for (auto it: up_to_date_tests) {
		if (it->name == current_test->name) {
			//already have that one
			return;
		}
	}

	for (auto it: passed_tests) {
		if (it->name == current_test->name) {
			//already have that one
			return;
		}
	}

	passed_tests.push_back(current_test);
	current_test = nullptr;
}

void Reporter::test_failed() {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();
	auto duration = duration_to_str(current_test->stop_timestamp - current_test->start_timestamp);

	report(fmt::format("{} Test ", progress()), red, true);
	report(current_test->name, yellow, true);
	report(fmt::format(" FAILED in {}\n", duration), red, true);

	bool already_failed = false;
	for (auto it: failed_tests) {
		if (it->name == current_test->name) {
			already_failed = true;
		}
	}

	if (!already_failed) {
		failed_tests.push_back(current_test);
	}
	current_test = nullptr;
}

void Reporter::print_statistics()
{
	auto tests_durantion = duration_to_str(std::chrono::system_clock::now() - start_timestamp);
	auto total_tests = passed_tests.size() + failed_tests.size() + up_to_date_tests.size() + ignored_tests.size();

	report(fmt::format("PROCESSED TOTAL {} TESTS IN {}\n", total_tests, tests_durantion), blue, true);
	report(fmt::format("UP-TO-DATE: {}\n", up_to_date_tests.size()), blue, true);

	if (ignored_tests.size()) {
		report(fmt::format("LOST CACHE, BUT SKIPPED: {}\n", ignored_tests.size()), yellow, true);
		for (auto ignore: ignored_tests) {
			report(fmt::format("\t -{}\n", ignore->name), yellow);
		}
	}
	report(fmt::format("RUN SUCCESSFULLY: {}\n", passed_tests.size()), green, true);
	report(fmt::format("FAILED: {}\n", failed_tests.size()), red, true);
	for (auto fail: failed_tests) {
		report(fmt::format("\t -{}\n", fail->name), red);
	}
}

void Reporter::create_controller(std::shared_ptr<Controller> controller) {
	report(fmt::format("{} Creating {} ", progress(), controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::take_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) {
	report(fmt::format("{} Taking snapshot ", progress()), blue);
	report(snapshot, yellow);
	report(fmt::format(" for {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::restore_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) {
	report(fmt::format("{} Restoring snapshot ", progress()), blue);
	report(snapshot, yellow);
	report(fmt::format(" for {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::print(std::shared_ptr<VmController> vmc, const std::string& message) {
	report(progress() + " ", blue);
	report(vmc->name(), yellow);
	report(fmt::format(" :{}\n", message), blue);
}

void Reporter::start(std::shared_ptr<VmController> vmc) {
	report(fmt::format("{} Starting virtual machine ", progress()), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::stop(std::shared_ptr<VmController> vmc) {
	report(fmt::format("{} Stopping virtual machine ", progress()), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::shutdown(std::shared_ptr<VmController> vmc, const std::string& timeout) {
	report(fmt::format("{} Shutting down virtual machine ", progress()), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", timeout), blue);
}

void Reporter::press_key(std::shared_ptr<VmController> vmc, const std::string& key, uint32_t times) {
	report(fmt::format("{} Pressing key ", progress()), blue);
	report(fmt::format("{} ", key), yellow);

	if (times > 1) {
		report(fmt::format("{} times ", times), blue);
	}

	report("on virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::type(std::shared_ptr<VmController> vmc, const std::string& text) {
	report(fmt::format("{} Typing ", progress()), blue);
	report(fmt::format("\"{}\" ", text), yellow);
	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);

}

void Reporter::sleep(std::shared_ptr<VmController> vmc, const std::string& timeout) {
	report(fmt::format("{} Sleeping in virtual machine ", progress()), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" for {}\n", timeout), blue);
}

void Reporter::wait(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) {
	report(fmt::format("{} Waiting ", progress()), blue);
	report(fmt::format("{} ", text), yellow);
	report(fmt::format("for {} in virtual machine ", timeout), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::check(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) {
	report(fmt::format("{} Checking ", progress()), blue);
	report(fmt::format("{} ", text), yellow);
	if (timeout != "1ms") {
		report(fmt::format(" for {} ", timeout), blue);
	}
	report(fmt::format("in virtual machine "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::macro_call(std::shared_ptr<VmController> vmc, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params) {
	report(fmt::format("{} Calling macro ", progress()), blue);
	report(fmt::format("{}(", macro_name), yellow);

	for (auto it = params.begin(); it != params.end(); ++it) {
		report(fmt::format("{}=\"{}\"", it->first, it->second), yellow);

		if ((it + 1) != params.end()) {
			report(", ", yellow);
		}
	}

	report(")", yellow);
	report(" in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::plug(std::shared_ptr<VmController> vmc, const std::string& device, const std::string& device_name, bool is_on) {
	std::string plug_or_unplug = is_on ? "Plugging" : "Unplugging";
	std::string into_or_from = is_on ? "into" : "from";
	report(fmt::format("{} {} {} ", progress(), plug_or_unplug, device), blue);
	report(fmt::format("{} ", device_name), yellow);
	report(fmt::format("{} virtual machine ", into_or_from), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::exec(std::shared_ptr<VmController> vmc, const std::string& interpreter, const std::string& timeout) {
	report(fmt::format("{} Executing {} command in virtual machine ", progress(), interpreter), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", timeout), blue);
}

void Reporter::copy(std::shared_ptr<VmController> vmc, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout) {
	std::string from_to = is_to_guest ? "to" : "from";

	report(fmt::format("{} Copying ", progress()), blue);
	report(fmt::format("{} ", from), yellow);
	report(fmt::format("{} virtual machine ", from_to), blue);
	report(fmt::format("{} ", vmc->name()), yellow);
	report(fmt::format("to destination "), blue);
	report(fmt::format("{} ", to), yellow);
	report(fmt::format(" with timeout {}\n", timeout), blue);
}

void Reporter::mouse_move(std::shared_ptr<VmController> vmc, const std::string& X, const std::string& Y) {
	report(fmt::format("{} Moving cursor ", progress()), blue);
	report(fmt::format("X: {} Y: {} ", X, Y), yellow);
	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_click(std::shared_ptr<VmController> vmc, const std::string& click_type) {
	report(fmt::format("{} {} on virtual machine ", progress(), click_type), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::exec_command_output(const std::string& text) {
	report(text, regular);
}

nlohmann::json Reporter::create_json_report() const {
	nlohmann::json report = nlohmann::json::object();
	report["tests"] = nlohmann::json::array();

	for (auto test: passed_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name},
			{"description", test->description},
			{"status", "success"},
			{"is_cached", false},
			{"duration", std::chrono::duration_cast<std::chrono::seconds>(duration).count()}
		};

		report["tests"].push_back(test_json);
	}

	for (auto test: failed_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name},
			{"description", test->description},
			{"status", "fail"},
			{"is_cached", false},
			{"duration", std::chrono::duration_cast<std::chrono::seconds>(duration).count()}
		};

		report["tests"].push_back(test_json);
	}

	for (auto test: up_to_date_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name},
			{"description", test->description},
			{"status", "success"},
			{"is_cached", true},
			{"duration", std::chrono::duration_cast<std::chrono::seconds>(duration).count()}
		};

		report["tests"].push_back(test_json);
	}

	auto start_timestamp_t = std::chrono::system_clock::to_time_t(start_timestamp);

	std::stringstream ss1;
	ss1 << std::put_time(std::localtime(&start_timestamp_t), "%FT%T%z");
	report["start_timestamp"] = ss1.str();

	auto stop_timestamp_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	std::stringstream ss2;
	ss2 << std::put_time(std::localtime(&stop_timestamp_t), "%FT%T%z");

	report["stop_timestamp"] = ss2.str();

	return report;
}

void Reporter::report(const std::string& message, style color, bool is_bold) {
	std::cout << rang::style::reset;

	switch (color) {
		case regular:
			break;
		case blue:
			std::cout << rang::fgB::blue;
			break;
		case magenta:
			std::cout << rang::fgB::magenta;
			break;
		case yellow:
			std::cout << rang::fg::yellow;
			break;
		case green:
			std::cout << rang::fg::green;
			break;
		case red:
			std::cout << rang::fg::red;
			break;
	}

	if (is_bold) {
		std::cout << rang::style::bold;
	}

	std::cout << message;

	if (current_test) {
		current_test->output_file << message;
	} else {
		summary_output_file << message;
	}
}
