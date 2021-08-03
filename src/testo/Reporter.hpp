
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "IR/Test.hpp"

struct ReporterConfig {
	std::string report_folder;
	bool html = false;

	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};

struct Reporter {
	Reporter() = default;
	Reporter(const ReporterConfig& config);

	void init(const std::vector<std::shared_ptr<IR::TestRun>>& _tests_runs,
		const std::vector<std::shared_ptr<IR::Test>>& _up_to_date_tests);

	void finish();

	//test stuff
	void skip_test();
	void prepare_environment();
	void run_test();
	void test_passed();
	void test_failed(const std::string& error_message);

	void print_statistics();

	//Controller stuff
	void create_controller(std::shared_ptr<IR::Controller> controller);
	void take_snapshot(std::shared_ptr<IR::Controller> controller, const std::string& snapshot);
	void restore_snapshot(std::shared_ptr<IR::Controller> controller, const std::string& snapshot);

	//both controller actions
	void print(std::shared_ptr<IR::Controller> controller, const std::string& message);
	void sleep(std::shared_ptr<IR::Controller> controller, const std::string& timeout);
	void macro_action_call(std::shared_ptr<IR::Controller> controller, const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params);
	void macro_command_call(const std::string& macro_name, const std::vector<std::pair<std::string, std::string>>& params);

	//vm actions
	void start(std::shared_ptr<IR::Machine> vmc);
	void stop(std::shared_ptr<IR::Machine> vmc);
	void shutdown(std::shared_ptr<IR::Machine> vmc, const std::string& timeout);
	void press_key(std::shared_ptr<IR::Machine> vmc, const std::string& key, uint32_t times);
	void hold_key(std::shared_ptr<IR::Machine> vmc, const std::string& key);
	void release_key(std::shared_ptr<IR::Machine> vmc, const std::string& key);
	void release_key(std::shared_ptr<IR::Machine> vmc);
	void type(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& interval);
	void wait(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& timeout, const std::string& interval);
	void check(std::shared_ptr<IR::Machine> vmc, const std::string& text, const std::string& timeout, const std::string& interval);
	void plug(std::shared_ptr<IR::Machine> vmc, const std::string& device, const std::string& device_name, bool is_on);
	void exec(std::shared_ptr<IR::Machine> vmc, const std::string& interpreter, const std::string& timeout);
	void copy(std::shared_ptr<IR::Controller> controller, const std::string& from, const std::string& to, bool is_to_guest, const std::string& timeout);
	void screenshot(std::shared_ptr<IR::Machine> controller, const std::string& destination);
	void mouse_move_click_coordinates(std::shared_ptr<IR::Machine> vmc, const std::string& x, const std::string& y);
	void mouse_move_click_selectable(std::shared_ptr<IR::Machine> vmc, const std::string& object, const std::string& timeout);
	void mouse_no_object();
	void mouse_move_click(std::shared_ptr<IR::Machine> vmc, std::string event);
	void mouse_hold(std::shared_ptr<IR::Machine> vmc, std::string button);
	void mouse_release(std::shared_ptr<IR::Machine> vmc);
	void mouse_wheel(std::shared_ptr<IR::Machine> vmc, const std::string& direction);

	//negotiator
	void exec_command_output(const std::string& text);

	void save_screenshot(std::shared_ptr<IR::Machine> vmc, const stb::Image<stb::RGB>& screenshot);

	std::string progress() const {
		std::stringstream ss;
		ss << "[";
		ss << std::setw(3);
		ss << std::round(current_progress());
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
	void report_prefix(style color, bool is_bold = false);

	void print_stdout(const std::string& message, style color, bool is_bold);
	void print_stdout_html(const std::string& message, style color, bool is_bold);
	void print_stdout_terminal(const std::string& message, style color, bool is_bold);
	void print_file(const std::string& message);

	std::vector<std::shared_ptr<IR::TestRun>> tests_runs;
	std::vector<std::shared_ptr<IR::Test>> up_to_date_tests;

	std::map<std::string, size_t> get_stats(IR::TestRun::ExecStatus status) const;

private:
	std::shared_ptr<IR::TestRun> current_test_run;
	size_t current_test_run_index = 0;

	float current_progress() const;
	nlohmann::json create_json_report() const;

	std::chrono::system_clock::time_point start_timestamp;

	bool html;

	fs::path report_folder;
	std::string launch_id = generate_uuid_v4();
	std::ofstream output_file;
	nlohmann::json config = nlohmann::json::object();
};
