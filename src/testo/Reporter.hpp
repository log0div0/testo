
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>
#include "backends/VmController.hpp"
#include "Node.hpp"

struct Reporter {
	struct Test {
		Test() = default;
		Test(std::shared_ptr<AST::Test> test): name(test->name), description(test->description) {}
		std::string name;
		std::string description;
		std::chrono::system_clock::time_point start_timestamp;
		std::chrono::system_clock::time_point stop_timestamp;
	};

	Reporter() = delete;
	Reporter(const nlohmann::json& config);

	void init(const std::list<std::shared_ptr<AST::Test>>& _tests_to_run,
		const std::vector<std::shared_ptr<AST::Test>>& _up_to_date_tests,
		const std::vector<std::shared_ptr<AST::Test>>& _ignored_tests);

	void finish();

	//test stuff
	void prepare_environment();
	void run_test();
	void skip_failed_test(const std::string& failed_parent);
	void test_passed();
	void test_failed();

	void print_statistics();
	nlohmann::json create_json_report() const;

	//Controller stuff
	void create_controller(std::shared_ptr<Controller> controller) const;
	void take_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) const;
	void restore_snapshot(std::shared_ptr<Controller> controller, const std::string& snapshot) const;

	//vm actions
	void print(std::shared_ptr<VmController> vmc, const std::string& message) const;
	void start(std::shared_ptr<VmController> vmc) const;
	void stop(std::shared_ptr<VmController> vmc) const;
	void shutdown(std::shared_ptr<VmController> vmc, const std::string& timeout) const;
	void press_key(std::shared_ptr<VmController> vmc, const std::string& key, uint32_t times) const;
	void type(std::shared_ptr<VmController> vmc, const std::string& text) const;
	void sleep(std::shared_ptr<VmController> vmc, const std::string& timeout) const;
	void wait(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) const;
	void check(std::shared_ptr<VmController> vmc, const std::string& text, const std::string& timeout) const;
	void macro_call(std::shared_ptr<VmController> vmc, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params) const;
	void plug(std::shared_ptr<VmController> vmc, const std::string& device, const std::string& device_name, bool is_on) const;
	void exec(std::shared_ptr<VmController> vmc, const std::string& interpreter, const std::string& timeout) const;
	void copy(std::shared_ptr<VmController> vmc, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout) const;
	void mouse_move(std::shared_ptr<VmController> vmc, const std::string& X, const std::string& Y) const;
	void mouse_click(std::shared_ptr<VmController> vmc, const std::string& click_type) const;

	std::string progress() const {
		std::stringstream ss;
		ss << "[";
		ss << std::setw(3);
		ss << std::round(current_progress);
		ss << std::setw(0);
		ss << '%' << "]";
		return ss.str();
	}

	std::string prefix;

	std::list<std::shared_ptr<Test>> tests_to_run;
	std::vector<std::shared_ptr<Test>> passed_tests;
	std::vector<std::shared_ptr<Test>> failed_tests;
	std::vector<std::shared_ptr<Test>> up_to_date_tests;
	std::vector<std::shared_ptr<Test>> ignored_tests;

	std::shared_ptr<Test> current_test;

	float progress_step = 0;
	float current_progress = 0;

	std::chrono::system_clock::time_point start_timestamp;
	std::chrono::system_clock::time_point finish_timestamp;

	fs::path report_folder() const;
};

