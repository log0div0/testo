
#include "Logger.hpp"
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

Logger::Logger(const nlohmann::json& config) {
	json_report_file = config.at("json_report_file").get<std::string>();
}

void Logger::init(const std::vector<std::string>& _tests_to_run, const std::vector<std::string>& _up_to_date_tests, const std::vector<std::string>& _ignored_tests) {
	for (auto test: _tests_to_run) {
		tests_to_run.push_back(std::shared_ptr<Test>(new Test(test)));
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
		std::cout << rang::fgB::blue << rang::style::bold;
		std::cout << "UP-TO-DATE TESTS:" << std::endl;
		std::cout << rang::style::reset;
		std::cout << rang::fgB::magenta;
		for (auto test: up_to_date_tests) {
			current_progress += progress_step;
			std::cout << test->name << std::endl;
		}
		std::cout << rang::style::reset;
	}

	if (tests_to_run.size()) {
		std::cout << rang::fgB::blue << rang::style::bold;
		std::cout << "TESTS TO RUN:" << std::endl;
		std::cout << rang::style::reset;
		std::cout << rang::fgB::magenta;
		for (auto test: tests_to_run) {
			std::cout << test->name << std::endl;
		}
		std::cout << rang::style::reset;
	}

	start_timestamp = std::chrono::system_clock::now();
}

void Logger::finish() {
	finish_timestamp = std::chrono::system_clock::now();

	print_statistics();
	if (json_report_file.length()) {
		auto path = fs::absolute(json_report_file);
		auto report = create_json_report();

		fs::create_directories(path.parent_path());

		std::ofstream file(path);
		file << report.dump(2);
	}
}

void Logger::prepare_environment() {
	current_test = tests_to_run.front();
	tests_to_run.pop_front();

	std::cout
		<< rang::fgB::blue << progress()
		<< " Preparing the environment for test "
		<< rang::fg::yellow << current_test->name
		<< rang::style::reset << std::endl;

	current_test->start_timestamp = std::chrono::system_clock::now();
}

void Logger::run_test() {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Running test "
		<< rang::fg::yellow << current_test->name
		<< rang::style::reset << std::endl;
}

void Logger::skip_failed_test(const std::string& failed_parent) {
	std::cout
		<< rang::fgB::red << progress()
		<< " Skipping test "
		<< rang::fg::yellow << current_test->name
		<< rang::fgB::red << " because his parent "
		<< rang::fg::yellow << failed_parent
		<< rang::fgB::red << " failed"
		<< rang::style::reset << std::endl;

	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();
	failed_tests.push_back(current_test);
	current_test = nullptr;
}

void Logger::test_passed() {
	current_test->stop_timestamp = std::chrono::system_clock::now();
	auto duration = duration_to_str(current_test->stop_timestamp - current_test->start_timestamp);
	std::cout
		<< rang::fgB::green << progress()
		<< " Test " << rang::fg::yellow << current_test->name
		<< rang::fgB::green << " PASSED in "
		<< duration
		<< rang::style::reset << std::endl;

	current_progress += progress_step;

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

void Logger::test_failed() {
	current_progress += progress_step;
	current_test->stop_timestamp = std::chrono::system_clock::now();

	std::cout
		<< rang::fgB::red << progress()
		<< " Test "
		<< rang::fg::yellow << current_test->name
		<< rang::fgB::red << " FAILED in "
		//<< time
		<< rang::style::reset << std::endl;

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

void Logger::print_statistics()
{
	auto tests_durantion = duration_to_str(finish_timestamp - start_timestamp);
	/*auto total_tests = succeeded_tests.size() + failed_tests.size() + up_to_date_tests.size() + ignored_tests.size();

	std::cout << rang::style::bold;
	std::cout << rang::fg::blue;
	std::cout << "PROCESSED TOTAL " << total_tests << " TESTS IN " << time << std::endl;
	std::cout << "UP-TO-DATE: " << up_to_date_tests.size() << std::endl;
	if (ignored_tests.size()) {
		std::cout << "LOST CACHE, BUT SKIPPED: " << ignored_tests.size() << std::endl;
		for (auto ignore: ignored_tests) {
			std::cout << "\t -" << ignore->name.value() << std::endl;
		}
	}
	std::cout << rang::fg::green;
	std::cout << "RUN SUCCESSFULLY: " << succeeded_tests.size() << std::endl;
	std::cout << rang::fg::red;
	std::cout << "FAILED: " << failed_tests.size() << std::endl;
	std::cout << rang::style::reset;
	std::cout << rang::fg::red;
	for (auto fail: failed_tests) {
		std::cout << "\t -" << fail->name.value() << std::endl;
	}
	std::cout << rang::style::reset;*/
}

void Logger::create_controller(std::shared_ptr<Controller> controller) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Creating " << controller->type() << " "
		<< rang::fg::yellow << controller->name()
		<< rang::style::reset << std::endl;
}

void Logger::take_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Taking snapshot "
		<< rang::fg::yellow << snapshot
		<< rang::fgB::blue << " for " << controller->type() << " "
		<< rang::fg::yellow << controller->name()
		<< rang::style::reset << std::endl;
}

void Logger::restore_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Restoring snapshot "
		<< rang::fg::yellow << snapshot
		<< rang::fgB::blue << " for " << controller->type() << " "
		<< rang::fg::yellow << controller->name()
		<< rang::style::reset << std::endl;
}

void Logger::print(std::shared_ptr<VmController> vmc, const std::string& message) const {
	std::cout
		<< rang::fgB::blue << progress() << " "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue << ": " << message
		<< rang::style::reset << std::endl;
}

void Logger::start(std::shared_ptr<VmController> vmc) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Starting virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::stop(std::shared_ptr<VmController> vmc) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Stopping virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::shutdown(std::shared_ptr<VmController> vmc, const std::string& timeout) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Shutting down virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue << " with timeout " << timeout
		<< rang::style::reset << std::endl;
}

void Logger::press_key(std::shared_ptr<VmController> vmc, const std::string& key, uint32_t times) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Pressing key "
		<< rang::fg::yellow << key
		<< rang::fgB::blue;

	if (times > 1) {
		std::cout << " " << times << " times";
	}

	std::cout
		<< " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::type(std::shared_ptr<VmController> vmc, const std::string& text) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Typing "
		<< rang::fg::yellow << "\"" << text << "\""
		<< rang::fgB::blue << " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::sleep(std::shared_ptr<VmController> vmc, const std::string& timeout) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Sleeping "
		<< rang::fgB::blue << "in virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue
		<< " for " << timeout << rang::style::reset << std::endl;
}

void Logger::wait(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Waiting "
		<< rang::fg::yellow << text
		<< rang::fgB::blue << " for " << timeout
		<< " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue
		<< rang::style::reset << std::endl;
}

void Logger::check(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Checking "
		<< rang::fg::yellow
		<< text
		<< rang::fgB::blue;

	if (timeout != "1ms") {
		std::cout << " for " << timeout;
	}

	std::cout
		<< " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::macro_call(std::shared_ptr<VmController> vmc, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Calling macro "
		<< rang::fg::yellow << macro_name << "(";

	for (auto& pair: params) {
		std::cout
			<< pair.first << "=" << pair.second;
	}

	std::cout << ")";

	std::cout
		<< rang::fgB::blue << " in virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::plug(std::shared_ptr<VmController> vmc, const std::string& device, const std::string& device_name, bool is_on) const {
	std::string plug_or_unplug = is_on ? " Plugging " : " Unplugging ";
	std::string into_or_from = is_on ? " into " : " from ";
	std::cout
		<< rang::fgB::blue << progress()
		<< plug_or_unplug  << device
		<< rang::fg::yellow << " " << device_name
		<< rang::fgB::blue << into_or_from << "virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::exec(std::shared_ptr<VmController> vmc, const std::string& interpreter, const std::string& timeout) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Executing " << interpreter << " command in virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue << " with timeout " << timeout
		<< rang::style::reset << std::endl;
}

void Logger::copy(std::shared_ptr<VmController> vmc, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout) const {
	std::string from_to = is_to_guest ? "to" : "from";

	std::cout
		<< rang::fgB::blue << progress() << " Copying "
		<< rang::fg::yellow << from
		<< rang::fgB::blue << " " << from_to << " virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::fgB::blue << " in directory "
		<< rang::fg::yellow << to
		<< rang::fgB::blue << " with timeout " << timeout
		<< rang::style::reset << std::endl;
}

void Logger::mouse_move(std::shared_ptr<VmController> vmc, const std::string& X, const std::string& Y) const {
	std::cout
		<< rang::fgB::blue << progress()
		<< " Moving cursor "
		<< rang::fg::yellow << "X:" << X
		<< " Y:" << Y
		<< rang::fgB::blue << " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

void Logger::mouse_click(std::shared_ptr<VmController> vmc, const std::string& click_type) const {
	std::cout
		<< rang::fgB::blue << progress() << " "
		<< click_type
		<< rang::fgB::blue << " on virtual machine "
		<< rang::fg::yellow << vmc->name()
		<< rang::style::reset << std::endl;
}

nlohmann::json Logger::create_json_report() const {
	return nlohmann::json();
	/*nlohmann::json report = nlohmann::json::object();
	report["tests"] = nlohmann::json::array();

	for (auto test: succeeded_tests) {
		auto duration = test->stop_timestamp - test->start_timestamp;
		nlohmann::json test_json = {
			{"name", test->name.value()},
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
			{"name", test->name.value()},
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
			{"name", test->name.value()},
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

	return report;*/
}
