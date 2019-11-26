
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>
#include "backends/VmController.hpp"
#include "Node.hpp"

struct Logger {
	Logger() = delete;
	Logger(const nlohmann::json& config);

	//test stuff
	void prepare_environment(const std::string& test) const;
	void run_test(const std::string& test) const;
	void skip_test(const std::string& test, const std::string& parent) const;
	void test_passed(const std::string& test, const std::string& time) const;
	void test_failed(const std::string& test, const std::string& time) const;

	void print_statistics(
		const std::vector<std::shared_ptr<AST::Test>>& succeeded_tests,
		const std::vector<std::shared_ptr<AST::Test>>& failed_tests,
		const std::vector<std::shared_ptr<AST::Test>>& up_to_date_tests,
		const std::vector<std::shared_ptr<AST::Test>>& ignored_tests,
		const std::string& time);

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

	float current_progress = 0;
};

