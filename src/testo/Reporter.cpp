
#include "Reporter.hpp"
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

void ReporterConfig::validate() const {
}

Reporter::Reporter(const ReporterConfig& config) {
	report_folder = config.report_folder;
	html = config.html;
}

void Reporter::init(const std::list<std::shared_ptr<IR::Test>>& _tests_to_run,	const std::vector<std::shared_ptr<IR::Test>>& _up_to_date_tests)
{
	start_timestamp = std::chrono::system_clock::now();

	if (!report_folder.empty()) {
		if (fs::exists(report_folder)) {
			if (!fs::is_directory(report_folder)) {
				throw std::runtime_error("Specified report folder " + report_folder.generic_string() + " is not a folder");
			}
			if (!fs::is_empty(report_folder)) {
				throw std::runtime_error("Specified report folder " + report_folder.generic_string() + " is not empty");
			}
		}
		fs::create_directories(report_folder);
		output_file = std::ofstream(report_folder / "log.txt");
	}

	for (auto test: _tests_to_run) {
		if (!report_folder.empty()) {
			fs::create_directories(report_folder / test->name());
			test->output_file = std::ofstream(report_folder / test->name() / "log.txt");
		}
		tests_to_run.push_back(test);
	}
	for (auto test: _up_to_date_tests) {
		up_to_date_tests.push_back(test);
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
			report(fmt::format("{}\n", test->name()), magenta);
		}
	}

	if (tests_to_run.size()) {
		report("TESTS TO RUN:\n", style::blue, true);
		for (auto test: tests_to_run) {
			report(fmt::format("{}\n", test->name()), magenta);
		}
	}
}

void Reporter::finish() {
	finish_timestamp = std::chrono::system_clock::now();

	print_statistics();
	if (!report_folder.empty()) {
		auto json_report_file = report_folder / "report.json";

		auto path = fs::absolute(json_report_file);
		auto report = create_json_report();
		fs::create_directories(path.parent_path());
		std::ofstream file(path);
		file << report.dump(2);
	}
}

void Reporter::prepare_environment() {
	current_test = tests_to_run.front();
	tests_to_run.pop_front();

	report_prefix(blue);
	report(fmt::format("Preparing the environment for test "), blue);
	report(fmt::format("{}\n", current_test->name()), yellow);

	current_test->start_timestamp = std::chrono::system_clock::now();
}

void Reporter::run_test() {
	report_prefix(blue);
	report(fmt::format("Running test "), blue);
	report(fmt::format("{}\n", current_test->name()), yellow);
}

void Reporter::skip_failed_test(const std::string& failed_parent) {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();

	report_prefix(red, true);
	report(fmt::format("Skipping test "), red, true);
	report(current_test->name(), yellow, true);
	report(" because his parent ", red, true);
	report(failed_parent, yellow, true);
	report(" failed\n", red, true);

	failed_tests.push_back(current_test);
	current_test = nullptr;
}

void Reporter::test_passed() {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();
	auto duration = duration_to_str(current_test->stop_timestamp - current_test->start_timestamp);

	report_prefix(green, true);
	report(fmt::format("Test "), green, true);
	report(current_test->name(), yellow, true);
	report(fmt::format(" PASSED in {}\n", duration), green, true);

	for (auto it: up_to_date_tests) {
		if (it->name() == current_test->name()) {
			//already have that one
			return;
		}
	}

	for (auto it: passed_tests) {
		if (it->name() == current_test->name()) {
			//already have that one
			return;
		}
	}

	passed_tests.push_back(current_test);
	current_test = nullptr;
}

void Reporter::test_failed(const std::string& error_message) {
	report(fmt::format("{}", error_message), red, true);

	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();
	auto duration = duration_to_str(current_test->stop_timestamp - current_test->start_timestamp);

	report_prefix(red, true);
	report(fmt::format("Test "), red, true);
	report(current_test->name(), yellow, true);
	report(fmt::format(" FAILED in {}\n", duration), red, true);

	bool already_failed = false;
	for (auto it: failed_tests) {
		if (it->name() == current_test->name()) {
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
	auto total_tests = passed_tests.size() + failed_tests.size() + up_to_date_tests.size();

	report(fmt::format("PROCESSED TOTAL {} TESTS IN {}\n", total_tests, tests_durantion), blue, true);
	report(fmt::format("UP-TO-DATE: {}\n", up_to_date_tests.size()), blue, true);

	report(fmt::format("RUN SUCCESSFULLY: {}\n", passed_tests.size()), green, true);
	report(fmt::format("FAILED: {}\n", failed_tests.size()), red, true);
	for (auto fail: failed_tests) {
		report(fmt::format("\t -{}\n", fail->name()), red);
	}
}

void Reporter::create_controller(std::shared_ptr<IR::Controller> controller) {
	report_prefix(blue);
	report(fmt::format("Creating {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::take_snapshot(std::shared_ptr<IR::Controller> controller, const std::string& snapshot) {
	report_prefix(blue);
	report(fmt::format("Taking snapshot "), blue);
	report(snapshot, yellow);
	report(fmt::format(" for {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::restore_snapshot(std::shared_ptr<IR::Controller> controller, const std::string& snapshot) {
	report_prefix(blue);
	report(fmt::format("Restoring snapshot "), blue);
	report(snapshot, yellow);
	report(fmt::format(" for {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::print(std::shared_ptr<IR::Controller> controller, const std::string& message) {
	report_prefix(blue);
	report(controller->name(), yellow);
	report(fmt::format(": {}\n", message), blue);
}

void Reporter::start(std::shared_ptr<IR::Machine> vmc) {
	report_prefix(blue);
	report(fmt::format("Starting virtual machine "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::stop(std::shared_ptr<IR::Machine> vmc) {
	report_prefix(blue);
	report(fmt::format("Stopping virtual machine "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::shutdown(std::shared_ptr<IR::Machine> vmc, const std::string& timeout) {
	report_prefix(blue);
	report(fmt::format("Shutting down virtual machine "), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", timeout), blue);
}

void Reporter::press_key(std::shared_ptr<IR::Machine> vmc, const std::string& key, uint32_t times) {
	report_prefix(blue);
	report(fmt::format("Pressing key "), blue);
	report(fmt::format("{} ", key), yellow);

	if (times > 1) {
		report(fmt::format("{} times ", times), blue);
	}

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::hold_key(std::shared_ptr<IR::Machine> vmc, const std::string& key) {
	report_prefix(blue);
	report(fmt::format("Holding key "), blue);
	report(fmt::format("{} ", key), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::release_key(std::shared_ptr<IR::Machine> vmc, const std::string& keys) {
	report_prefix(blue);
	report(fmt::format("Releasing key "), blue);
	report(fmt::format("{} ", keys), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::release_key(std::shared_ptr<IR::Machine> vmc) {
	report_prefix(blue);
	report(fmt::format("Releasing "), blue);
	report("all held ", yellow);

	report("keys in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::type(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& interval) {
	report_prefix(blue);
	report(fmt::format("Typing "), blue);
	report(fmt::format("\"{}\" ", text), yellow);
	report(fmt::format("with interval {} in virtual machine ", interval), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);

}

void Reporter::sleep(std::shared_ptr<IR::Controller> controller, const std::string& timeout) {
	report_prefix(blue);
	report(fmt::format("Sleeping in {} ", controller->type()), blue);
	report(fmt::format("{}", controller->name()), yellow);
	report(fmt::format(" for {}\n", timeout), blue);
}

void Reporter::macro_action_call(std::shared_ptr<IR::Controller> controller, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params) {
	report_prefix(blue);
	report(fmt::format("Calling macro "), blue);
	report(fmt::format("{}(", macro_name), yellow);

	for (auto it = params.begin(); it != params.end(); ++it) {
		report(fmt::format("{}=\"{}\"", it->first, it->second), yellow);

		if ((it + 1) != params.end()) {
			report(", ", yellow);
		}
	}

	report(")", yellow);
	report(fmt::format(" in {} ", controller->type()), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::macro_command_call(const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params) {
	report_prefix(blue);
	report(fmt::format("Calling command macro "), blue);
	report(fmt::format("{}(", macro_name), yellow);

	for (auto it = params.begin(); it != params.end(); ++it) {
		report(fmt::format("{}=\"{}\"", it->first, it->second), yellow);

		if ((it + 1) != params.end()) {
			report(", ", yellow);
		}
	}

	report(")\n", yellow);
}

void Reporter::wait(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& timeout, const std::string& interval) {
	report_prefix(blue);
	report(fmt::format("Waiting "), blue);
	report(fmt::format("{} ", text), yellow);
	report(fmt::format("for {} with interval {} in virtual machine ", timeout, interval), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::check(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& timeout, const std::string& interval) {
	report_prefix(blue);
	report(fmt::format("Checking "), blue);
	report(fmt::format("{}", text), yellow);
	if (timeout != "1ms") {
		report(fmt::format(" for {}", timeout), blue);
	}
	if (interval != "1s") {
		report(fmt::format(" with interval {}", interval), blue);
	}
	report(fmt::format(" in virtual machine "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::plug(std::shared_ptr<IR::Machine> vmc, const std::string& device, const std::string& device_name, bool is_on) {
	std::string plug_or_unplug = is_on ? "Plugging" : "Unplugging";
	std::string into_or_from = is_on ? "into" : "from";
	report_prefix(blue);
	report(fmt::format("{} {} ", plug_or_unplug, device), blue);
	report(fmt::format("{} ", device_name), yellow);
	report(fmt::format("{} virtual machine ", into_or_from), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::exec(std::shared_ptr<IR::Machine> vmc, const std::string& interpreter, const std::string& timeout) {
	report_prefix(blue);
	report(fmt::format("Executing {} command in virtual machine ", interpreter), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", timeout), blue);
}

void Reporter::copy(std::shared_ptr<IR::Controller> controller, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout) {
	std::string from_to = is_to_guest ? "to" : "from";

	report_prefix(blue);
	report(fmt::format("Copying "), blue);
	report(fmt::format("{} ", from), yellow);
	report(fmt::format("{} {} ", from_to, controller->type()), blue);
	report(fmt::format("{} ", controller->name()), yellow);
	report(fmt::format("to destination "), blue);
	report(fmt::format("{} ", to), yellow);
	report(fmt::format("with timeout {}\n", timeout), blue);
}

void Reporter::screenshot(std::shared_ptr<IR::Machine> controller, const std::string& destination) {
	report_prefix(blue);
	report(fmt::format("Saving screenshot "), blue);
	report(fmt::format("from virtual machine {} ", controller->name()), yellow);
	report(fmt::format("to destination "), blue);
	report(fmt::format("{}", destination), yellow);
}

void Reporter::mouse_move_click_coordinates(std::shared_ptr<IR::Machine> vmc, const std::string& x, const std::string& y) {
	report("on coordinates ", blue);
	report(fmt::format("{} {} ", x, y), yellow);
	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_move_click_selectable(std::shared_ptr<IR::Machine> vmc, const std::string& object, const std::string& timeout) {
	report("on ", blue);
	report(fmt::format("{} ", object), yellow);
	report(fmt::format("with timeout {} in virtual machine ", timeout), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_move_click(std::shared_ptr<IR::Machine> vmc, std::string event) {
	if (event == "move") {
		event = "moving";
	} else if (event == "click") {
		event = "clicking";
	} else if (event == "lclick") {
		event = "left-clicking";
	} else if (event == "rclick") {
		event = "right-clicking";
	} else if (event == "mclick") {
		event = "middle-clicking";
	} else if (event == "dclick") {
		event = "double-clicking";
	}

	report_prefix(blue);
	report(fmt::format("Mouse {} ", event), blue);
}

void Reporter::mouse_no_object() {
	report("\n", blue);
}

void Reporter::mouse_hold(std::shared_ptr<IR::Machine> vmc, std::string button) {
	if (button == "lbtn") {
		button = "left button";
	} else if (button == "rbtn") {
		button = "right button";
	} else if (button == "mbtn") {
		button = "middle button";
	}

	report_prefix(blue);
	report(fmt::format("Mouse hold the "), blue);
	report(fmt::format("{} ", button), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_release(std::shared_ptr<IR::Machine> vmc) {
	report_prefix(blue);
	report(fmt::format("Mouse release buttons "), blue);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_wheel(std::shared_ptr<IR::Machine> vmc, const std::string& direction) {
	report_prefix(blue);
	report(fmt::format("Mouse wheel "), blue);
	report(fmt::format("{} ", direction), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::exec_command_output(const std::string& text) {
	report(text, regular);
}

void Reporter::save_screenshot(std::shared_ptr<IR::Machine> vmc, const stb::Image<stb::RGB>& screenshot) {
	if (report_folder.empty()) {
		return;
	}
	screenshot.write_png((report_folder / current_test->name() / "screenshot.png").generic_string());
	report_prefix(blue);
	report(fmt::format("Saved screenshot from vm "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

nlohmann::json Reporter::create_json_report() const {
	nlohmann::json report = nlohmann::json::object();
	report["tests"] = nlohmann::json::array();

	for (auto test: passed_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name()},
			{"description", test->description()},
			{"status", "success"},
			{"is_cached", false},
			{"duration", std::chrono::duration_cast<std::chrono::seconds>(duration).count()}
		};

		report["tests"].push_back(test_json);
	}

	for (auto test: failed_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name()},
			{"description", test->description()},
			{"status", "fail"},
			{"is_cached", false},
			{"duration", std::chrono::duration_cast<std::chrono::seconds>(duration).count()}
		};

		report["tests"].push_back(test_json);
	}

	for (auto test: up_to_date_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name()},
			{"description", test->description()},
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

std::string newline_to_br(const std::string& str) {
	std::string::size_type pos = 0;
	std::stringstream ss;
	while (true)
	{
		std::string::size_type n = str.find("\n", pos);
		ss << str.substr(pos, n - pos);
		if (n == std::string::npos) {
			return ss.str();
		} else {
			ss << "<br/>";
		}
		pos = n + 1;
	}
}

void Reporter::report(const std::string& message, style color, bool is_bold) {
	print_stdout(message, color, is_bold);
	print_file(message);
}

void Reporter::report_prefix(style color, bool is_bold) {
	print_stdout(fmt::format("{} ", progress()), color, is_bold);
	if (current_test) {
		print_file(fmt::format("[{}] ", current_test->name()));
	} else {
		print_file(fmt::format("[???] "));
	}
}

void Reporter::print_stdout_html(const std::string& message, style color, bool is_bold) {
	std::cout << "<span className=\"";
	switch (color) {
		case regular:
			break;
		case blue:
			std::cout << "blue";
			break;
		case magenta:
			std::cout << "magenta";
			break;
		case yellow:
			std::cout << "yellow";
			break;
		case green:
			std::cout << "green";
			break;
		case red:
			std::cout << "red";
			break;
	}
	std::cout << " ";
	if (is_bold) {
		std::cout << "bold";
	}
	std::cout << "\">";
	std::cout << newline_to_br(message);
	std::cout << "</span>" << std::endl;
}

void Reporter::print_stdout(const std::string& message, style color, bool is_bold) {
	if (html) {
		print_stdout_html(message, color, is_bold);
	} else {
		print_stdout_terminal(message, color, is_bold);
	}
}

void Reporter::print_stdout_terminal(const std::string& message, style color, bool is_bold) {
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

	std::cout << rang::style::reset;
}

void Reporter::print_file(const std::string& message) {
	if (!report_folder.empty()) {
		if (current_test) {
			current_test->output_file << message;
		} else {
			output_file << message;
		}
	}
}