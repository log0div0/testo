
#pragma once
#include "IR/Test.hpp"
#include "IR/Action.hpp"
#include "IR/Command.hpp"
#include "IR/Expr.hpp"
#include "Reporter.hpp"
#include "TemplateLiterals.hpp"

struct VisitorInterpreterConfig: ReporterConfig {
	bool stop_on_fail = false;
	bool assume_yes = false;
	std::string invalidate;

	void validate() const;
};

struct VisitorInterpreter {
	VisitorInterpreter(const VisitorInterpreterConfig& config);

	void visit();
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::ICmd> cmd);
	void visit_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::MacroBodyCommand>& macro_body);
	void visit_regular_command(const IR::RegularCommand& regular_cmd);

	std::shared_ptr<StackNode> stack;

private:
	friend struct IR::MacroCall;

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

	void check_up_to_date_tests(std::list<std::shared_ptr<IR::Test>>& tests_queue);
	void resolve_tests(const std::list<std::shared_ptr<IR::Test>>& tests_queue);
	void update_progress();

	void stop_all_vms(std::shared_ptr<IR::Test> test);

	std::shared_ptr<IR::Controller> current_controller;
	std::shared_ptr<IR::Test> current_test;
	template_literals::Parser template_parser;
	Reporter reporter;
};
