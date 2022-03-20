
#include "ReportWriterNativeRemote.hpp"
#include "Reporter.hpp"
#include <rang.hpp>
#include <fmt/format.h>
#include "../Logger.hpp"
#include "ReportWriterNativeLocal.hpp"
#include "ReportWriterAllure.hpp"

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

Reporter::Reporter(const ReporterConfig& config) {
	TRACE();

	if (config.report_folder.size()) {
		switch (config.get_report_format()) {
			case ReportFormat::NativeLocal:
				report_writer = std::make_unique<ReportWriterNativeLocal>(config);
				break;
			case ReportFormat::NativeRemote:
				report_writer = std::make_unique<ReportWriterNativeRemote>(config);
				break;
			case ReportFormat::Allure:
				report_writer = std::make_unique<ReportWriterAllure>(config);
				break;
		}
	} else {
		report_writer = std::make_unique<ReportWriter>(config);
	}

	html = config.html;
}

Reporter::~Reporter() {
	TRACE();
}

void Reporter::init(const std::vector<std::shared_ptr<IR::Test>>& _tests, const std::vector<std::shared_ptr<IR::TestRun>>& _tests_runs)
{
	TRACE();

	start_timestamp = std::chrono::system_clock::now();

	report_writer->launch_begin(_tests, _tests_runs);

	for (auto test_run: _tests_runs) {
		tests_runs.push_back(test_run);
	}
	for (auto test: _tests) {
		if (test->is_up_to_date()) {
			up_to_date_tests.push_back(test);
		}
	}

	if (up_to_date_tests.size()) {
		report("UP-TO-DATE TESTS:\n", blue, true);
		for (auto test: up_to_date_tests) {
			report(fmt::format("{}\n", test->name()), magenta);
		}
	}

	if (tests_runs.size()) {
		report("TESTS TO RUN:\n", style::blue, true);
		for (auto test_run: tests_runs) {
			report(fmt::format("{}\n", test_run->test->name()), magenta);
		}
	}
}

void Reporter::finish() {
	TRACE();

	print_statistics();
	report_writer->launch_end();
}

void Reporter::prepare_environment() {
	current_test_run = tests_runs.at(current_test_run_index);
	current_test_run->start_timestamp = std::chrono::system_clock::now();

	report_writer->test_begin(current_test_run);

	report_prefix(blue);
	report(fmt::format("Preparing the environment for test "), blue);
	report(fmt::format("{}\n", current_test_run->test->name()), yellow);
}

void Reporter::run_test() {
	report_prefix(blue);
	report(fmt::format("Running test "), blue);
	report(fmt::format("{}\n", current_test_run->test->name()), yellow);
}

std::string join(const std::set<std::string>& set, const std::string& delimiter) {
	size_t i = 0;
	std::string result;
	for (auto& str: set) {
		if (i) {
			result += delimiter;
		}
		result += str;
		++i;
	}
	return result;
}

void Reporter::skip_test() {
	current_test_run = tests_runs.at(current_test_run_index);
	current_test_run->exec_status = IR::TestRun::ExecStatus::Skipped;

	report_writer->test_skip_begin(current_test_run);

	std::set<std::string> names = current_test_run->get_unsuccessful_parents_names();
	std::string singular = "parent";
	std::string plural = "parents";
	if (names.size() == 0) {
		names = current_test_run->unsuccessful_deps_names;
		singular = "dependency";
		plural = "dependencies";
	}

	report_prefix(red, true);
	report(fmt::format("Skipping test "), red, true);
	report(current_test_run->test->name(), yellow, true);
	if (names.size() > 1) {
		report(" because his " + plural + " ", red, true);
	} else {
		report(" because his " + singular + " ", red, true);
	}
	report(join(names, ", "), yellow, true);
	if (names.size() > 1) {
		report(" are failed or skipped\n", red, true);
	} else {
		report(" is failed or skipped\n", red, true);
	}

	report_writer->test_skip_end(current_test_run);

	current_test_run = nullptr;
	++current_test_run_index;
}

void Reporter::test_passed() {
	current_test_run->stop_timestamp = std::chrono::system_clock::now();
	current_test_run->exec_status = IR::TestRun::ExecStatus::Passed;

	report_prefix(green, true);
	report(fmt::format("Test "), green, true);
	report(current_test_run->test->name(), yellow, true);
	report(fmt::format(" PASSED in {}\n", duration_to_str(current_test_run->duration())), green, true);

	report_writer->test_end(current_test_run);

	current_test_run = nullptr;
	++current_test_run_index;
}

void Reporter::test_failed(const std::string& message, const std::string& stacktrace, const std::string& failure_category) {
	report_raw(fmt::format("{}", stacktrace), red, true);

	current_test_run->failure_message = message;
	current_test_run->failure_stacktrace = stacktrace;
	current_test_run->failure_category = failure_category;
	current_test_run->stop_timestamp = std::chrono::system_clock::now();
	current_test_run->exec_status = IR::TestRun::ExecStatus::Failed;

	report_prefix(red, true);
	report(fmt::format("Test "), red, true);
	report(current_test_run->test->name(), yellow, true);
	report(fmt::format(" FAILED in {}\n", duration_to_str(current_test_run->duration())), red, true);

	report_writer->test_end(current_test_run);

	current_test_run = nullptr;
	++current_test_run_index;
}

void Reporter::error(const std::string& message) {
	report_raw(fmt::format("{}", message), red, true);
}

void Reporter::print_statistics()
{
	TRACE();

	auto passed_tests = get_stats(IR::TestRun::ExecStatus::Passed);
	auto failed_tests = get_stats(IR::TestRun::ExecStatus::Failed);
	auto skipped_tests = get_stats(IR::TestRun::ExecStatus::Skipped);

	auto tests_durantion = duration_to_str(std::chrono::system_clock::now() - start_timestamp);
	auto total_tests = tests_runs.size() + up_to_date_tests.size();

	report(fmt::format("PROCESSED TOTAL {} TESTS IN {}\n", total_tests, tests_durantion), blue, true);
	report(fmt::format("UP-TO-DATE: {}\n", up_to_date_tests.size()), blue, true);

	report(fmt::format("RUN SUCCESSFULLY: {}\n", passed_tests.size()), green, true);
	report(fmt::format("FAILED: {}\n", failed_tests.size()), red, true);
	for (auto kv: failed_tests) {
		if (kv.second > 1) {
			report(fmt::format("\t - {} ({} times)\n", kv.first, kv.second), red);
		} else {
			report(fmt::format("\t - {}\n", kv.first), red);
		}
	}
	report(fmt::format("SKIPPED: {}\n", skipped_tests.size()), magenta, true);
	for (auto kv: skipped_tests) {
		if (kv.second > 1) {
			report(fmt::format("\t - {} ({} times)\n", kv.first, kv.second), magenta);
		} else {
			report(fmt::format("\t - {}\n", kv.first), magenta);
		}
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

void Reporter::print(std::shared_ptr<IR::Controller> controller, const IR::Print& action) {
	report_prefix(blue);
	report(controller->name(), yellow);
	report(fmt::format(": {}\n", action.message()), blue);
}

void Reporter::repl_begin(std::shared_ptr<IR::Controller> controller, const IR::REPL& action) {
	report_prefix(blue);
	report(fmt::format("Entering REPL mode on "), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::repl_end(std::shared_ptr<IR::Controller> controller, const IR::REPL& action) {
	report_prefix(blue);
	report(fmt::format("Leaving REPL mode on "), blue);
	report(fmt::format("{}\n", controller->name()), yellow);
}

void Reporter::abort(std::shared_ptr<IR::Controller> controller, const IR::Abort& action) {
	report_prefix(blue);
	report(fmt::format("Aborting with a message: {}\n", action.message()), blue);
	if (std::shared_ptr<IR::Machine> vmc = std::dynamic_pointer_cast<IR::Machine>(controller)) {
		if (vmc->vm()->state() == VmState::Running) {
			report_writer->report_screenshot(current_test_run, vmc->make_new_screenshot());
		}
	}
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

void Reporter::shutdown(std::shared_ptr<IR::Machine> vmc, const IR::Shutdown& action) {
	report_prefix(blue);
	report(fmt::format("Shutting down virtual machine "), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", action.timeout().str()), blue);
}

void Reporter::press_key(std::shared_ptr<IR::Machine> vmc, const IR::KeySpec& key_spec) {
	report_prefix(blue);
	report(fmt::format("Pressing key "), blue);
	report(fmt::format("{} ", key_spec.combination().to_string()), yellow);

	auto times = key_spec.times();

	if (times > 1) {
		report(fmt::format("{} times ", times), blue);
	}

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::hold_key(std::shared_ptr<IR::Machine> vmc, const IR::Hold& action) {
	report_prefix(blue);
	report(fmt::format("Holding key "), blue);
	report(fmt::format("{} ", action.combination().to_string()), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::release_key(std::shared_ptr<IR::Machine> vmc, const IR::Release& action) {
	report_prefix(blue);
	report(fmt::format("Releasing key "), blue);
	report(fmt::format("{} ", action.combination().to_string()), yellow);

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

void Reporter::type(std::shared_ptr<IR::Machine> vmc, const IR::Type& action) {
	report_prefix(blue);
	report(fmt::format("Typing "), blue);
	report(fmt::format("\"{}\" ", action.text().str()), yellow);
	report(fmt::format("with interval {} in virtual machine ", action.interval().str()), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);

}

void Reporter::sleep(std::shared_ptr<IR::Controller> controller, const IR::Sleep& action) {
	report_prefix(blue);
	report(fmt::format("Sleeping in {} ", controller->type()), blue);
	report(fmt::format("{}", controller->name()), yellow);
	report(fmt::format(" for {}\n", action.timeout().str()), blue);
}

void Reporter::macro_action_call(std::shared_ptr<IR::Controller> controller, const IR::MacroCall& macro_call) {
	report_prefix(blue);
	report(fmt::format("Calling macro "), blue);
	report(fmt::format("{}(", macro_call.macro_name()), yellow);

	auto params = macro_call.args();

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

void Reporter::macro_command_call(const IR::MacroCall& macro_call) {
	report_prefix(blue);
	report(fmt::format("Calling command macro "), blue);
	report(fmt::format("{}(", macro_call.macro_name()), yellow);

	auto params = macro_call.args();

	for (auto it = params.begin(); it != params.end(); ++it) {
		report(fmt::format("{}=\"{}\"", it->first, it->second), yellow);

		if ((it + 1) != params.end()) {
			report(", ", yellow);
		}
	}

	report(")\n", yellow);
}

void Reporter::wait(std::shared_ptr<IR::Machine> vmc, const IR::Wait& action) {
	report_prefix(blue);
	report(fmt::format("Waiting "), blue);
	report(fmt::format("{} ", action.select_expr().to_string()), yellow);
	report(fmt::format("for {} with interval {} in virtual machine ", action.timeout().str(), action.interval().str()), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::check(std::shared_ptr<IR::Machine> vmc, const IR::Check& action) {
	std::string timeout = action.timeout().str();
	std::string interval = action.interval().str();
	report_prefix(blue);
	report(fmt::format("Checking "), blue);
	report(fmt::format("{}", action.select_expr().to_string()), yellow);
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

void Reporter::exec(std::shared_ptr<IR::Machine> vmc, const IR::Exec& action) {
	report_prefix(blue);
	report(fmt::format("Executing {} command in virtual machine ", action.interpreter()), blue);
	report(fmt::format("{}", vmc->name()), yellow);
	report(fmt::format(" with timeout {}\n", action.timeout().str()), blue);
}

void Reporter::copy(std::shared_ptr<IR::Controller> controller, const IR::Copy& action) {
	std::string from_to = action.ast_node->is_to_guest() ? "to" : "from";

	report_prefix(blue);
	report(fmt::format("Copying "), blue);
	report(fmt::format("{} ", action.from()), yellow);
	report(fmt::format("{} {} ", from_to, controller->type()), blue);
	report(fmt::format("{} ", controller->name()), yellow);
	report(fmt::format("to destination "), blue);
	report(fmt::format("{} ", action.to()), yellow);
	report(fmt::format("with timeout {}\n", action.timeout().str()), blue);
}

void Reporter::screenshot(std::shared_ptr<IR::Machine> controller, const IR::Screenshot& action) {
	report_prefix(blue);
	report(fmt::format("Saving screenshot "), blue);
	report(fmt::format("from virtual machine {} ", controller->name()), yellow);
	report(fmt::format("to destination "), blue);
	report(fmt::format("{}", action.destination()), yellow);
}

void Reporter::mouse_move_click_coordinates(std::shared_ptr<IR::Machine> vmc, const IR::MouseCoordinates& coordinates) {
	report("on coordinates ", blue);
	report(fmt::format("{} {} ", coordinates.x(), coordinates.y()), yellow);
	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_move_click_selectable(std::shared_ptr<IR::Machine> vmc, const IR::MouseSelectable& selectable) {
	report("on ", blue);
	report(fmt::format("{} ", selectable.to_string()), yellow);
	report(fmt::format("with timeout {} in virtual machine ", selectable.timeout().str()), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::mouse_move_click(std::shared_ptr<IR::Machine> vmc, const IR::MouseMoveClick& action) {
	std::string event = action.event_type();

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

void Reporter::mouse_hold(std::shared_ptr<IR::Machine> vmc, const IR::MouseHold& action) {
	std::string button = ToString(action.button());

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

void Reporter::mouse_wheel(std::shared_ptr<IR::Machine> vmc, const IR::MouseWheel& action) {
	report_prefix(blue);
	report(fmt::format("Mouse wheel "), blue);
	report(fmt::format("{} ", action.direction()), yellow);

	report("in virtual machine ", blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
}

void Reporter::exec_command_output(const std::string& text) {
	report_raw(text, regular);
}

void Reporter::js_stdout(const std::string& _stdout) {
	report_raw(fmt::format("{}\n", _stdout), yellow);
}

void Reporter::save_screenshot(std::shared_ptr<IR::Machine> vmc, const stb::Image<stb::RGB>& screenshot) {
	report_prefix(blue);
	report(fmt::format("Saved screenshot from vm "), blue);
	report(fmt::format("{}\n", vmc->name()), yellow);
	report_writer->report_screenshot(current_test_run, screenshot);
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
	print(message, color, is_bold);
	report_writer->report(current_test_run, message);
}

void Reporter::report_raw(const std::string& message, style color, bool is_bold) {
	print(message, color, is_bold);
	report_writer->report_raw(current_test_run, message);
}

void Reporter::report_prefix(style color, bool is_bold) {
	print(fmt::format("{} ", progress()), color, is_bold);
	report_writer->report_prefix(current_test_run);
}

void Reporter::print_html(const std::string& message, style color, bool is_bold) {
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

void Reporter::print(const std::string& message, style color, bool is_bold) {
	if (html) {
		print_html(message, color, is_bold);
	} else {
		print_terminal(message, color, is_bold);
	}
}

void Reporter::print_terminal(const std::string& message, style color, bool is_bold) {
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

float Reporter::current_progress() const {
	size_t total_tests_count = tests_runs.size() + up_to_date_tests.size();
	if (total_tests_count == 0) {
		return 100.0f;
	}
	return float(current_test_run_index + up_to_date_tests.size()) / total_tests_count * 100;
}

std::map<std::string, size_t> Reporter::get_stats(IR::TestRun::ExecStatus status) const {
	TRACE();

	std::map<std::string, size_t> result;
	for (auto& test_run: tests_runs) {
		if (test_run->exec_status == status) {
			++result[test_run->test->name()];
		}
	}
	return result;
}
