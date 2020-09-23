
#pragma once
#include "IR/Test.hpp"
#include "IR/Action.hpp"
#include "IR/Command.hpp"
#include "IR/Expr.hpp"
#include "Reporter.hpp"
#include <nn/OCR.hpp>

struct VisitorInterpreter {
	VisitorInterpreter(const nlohmann::json& config);

	void visit();
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(const IR::Command& cmd);

	std::shared_ptr<StackNode> stack;

private:
	//settings
	bool stop_on_fail;
	bool assume_yes;
	std::string invalidate;

	std::list<std::shared_ptr<IR::Test>> tests_to_run;
	std::vector<std::shared_ptr<IR::Test>> up_to_date_tests;
	std::vector<std::shared_ptr<IR::Test>> cache_missed_tests;

	void setup_vars();
	void reset_cache();

	bool parent_is_ok(std::shared_ptr<IR::Test> test, std::shared_ptr<IR::Test> parent,
		std::list<std::shared_ptr<IR::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<IR::Test>>::reverse_iterator end);

	void build_test_plan(std::shared_ptr<IR::Test> test,
		std::list<std::shared_ptr<IR::Test>>& test_plan,
		std::list<std::shared_ptr<IR::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<IR::Test>>::reverse_iterator end);

	bool is_cached(std::shared_ptr<IR::Test> test) const;
	bool is_cache_miss(std::shared_ptr<IR::Test> test) const;
	void check_up_to_date_tests(std::list<std::shared_ptr<IR::Test>>& tests_queue);
	void resolve_tests(const std::list<std::shared_ptr<IR::Test>>& tests_queue);
	void update_progress();

	void stop_all_vms(std::shared_ptr<IR::Test> test);

	std::shared_ptr<IR::Test> current_test;

	Reporter reporter;
};
