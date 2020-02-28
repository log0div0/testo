
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>
#include <fstream>
#include "backends/VmController.hpp"
#include "Node.hpp"

struct Reporter {
	struct Test {
		Test() = default;
		Test(std::shared_ptr<AST::Test> test): name(test->name), description(test->description) {}
		Test(std::shared_ptr<AST::Test> test, fs::path report_folder): name(test->name), description(test->description), output_file(report_folder / test->name.value()) {}
		std::string name;
		std::string description;
		std::chrono::system_clock::time_point start_timestamp;
		std::chrono::system_clock::time_point stop_timestamp;
		std::ofstream output_file;
	};

	Reporter() = default;
	Reporter(const nlohmann::json& config);

	void init(const std::list<std::shared_ptr<AST::Test>>& _tests_to_run,
		const std::vector<std::shared_ptr<AST::Test>>& _up_to_date_tests);

	void finish();

	//test stuff
	void prepare_environment();
	void run_test();
	void skip_failed_test(const std::string& failed_parent);
	void test_passed();
	void test_failed(const std::string& error_message);

	void print_statistics();
	nlohmann::json create_json_report() const;

	//Controller stuff
	void create_controller(std::shared_ptr<Controller> controller);
	void take_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot);
	void restore_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot);

	//vm actions
	void print(std::shared_ptr<VmController> vmc, const std::string& message);
	void start(std::shared_ptr<VmController> vmc);
	void stop(std::shared_ptr<VmController> vmc);
	void shutdown(std::shared_ptr<VmController> vmc, const std::string& timeout);
	void press_key(std::shared_ptr<VmController> vmc, const std::string& key, uint32_t times);
	void type(std::shared_ptr<VmController> vmc, const std::string& text);
	void sleep(std::shared_ptr<VmController> vmc, const std::string& timeout);
	void wait(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout, const std::string& interval);
	void check(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout, const std::string& interval);
	void macro_call(std::shared_ptr<VmController> vmc, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params);
	void plug(std::shared_ptr<VmController> vmc, const std::string& device, const std::string& device_name, bool is_on);
	void exec(std::shared_ptr<VmController> vmc, const std::string& interpreter, const std::string& timeout);
	void copy(std::shared_ptr<VmController> vmc, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout);
	void mouse_move_click(std::shared_ptr<VmController> vmc, std::string event, std::string object, const std::string& timeout);
	void mouse_hold(std::shared_ptr<VmController> vmc, std::string button);
	void mouse_release(std::shared_ptr<VmController> vmc);
	void mouse_wheel(std::shared_ptr<VmController> vmc, const std::string& direction);

	//negotiator
	void exec_command_output(const std::string& text);

	void save_screenshot(std::shared_ptr<VmController> vmc);

	std::string progress() const {
		std::stringstream ss;
		ss << "[";
		ss << std::setw(3);
		ss << std::round(current_progress);
		ss << std::setw(0);
		ss << '%' << "]";
		return ss.str();
	}


	enum style {
		regular,
		blue,
		magenta,
		yellow,
		green,
		red
	};

	void report(const std::string& message, style color, bool is_bold = false);

	std::list<std::shared_ptr<Test>> tests_to_run;
	std::vector<std::shared_ptr<Test>> passed_tests;
	std::vector<std::shared_ptr<Test>> failed_tests;
	std::vector<std::shared_ptr<Test>> up_to_date_tests;

	std::shared_ptr<Test> current_test;

	float progress_step = 0;
	float current_progress = 0;

	std::chrono::system_clock::time_point start_timestamp;
	std::chrono::system_clock::time_point finish_timestamp;

	bool report_logs;
	bool report_screenshots;

	fs::path report_folder;
	std::ofstream summary_output_file;
};

extern Reporter reporter;
