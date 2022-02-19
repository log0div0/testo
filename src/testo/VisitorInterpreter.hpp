
#pragma once
#include "IR/Test.hpp"
#include "IR/Action.hpp"
#include "IR/Command.hpp"
#include "IR/Expr.hpp"
#include "IR/Macro.hpp"
#include "Reporter.hpp"
#include "TemplateLiterals.hpp"
#include "Configs.hpp"

struct VisitorInterpreter {
	VisitorInterpreter(const VisitorInterpreterConfig& config);
	~VisitorInterpreter();

	void visit();
	void visit_test(const std::shared_ptr<IR::Test>& test);
	void visit_command_block(const std::shared_ptr<AST::Block<AST::Cmd>>& block);
	void visit_command(const std::shared_ptr<AST::Cmd>& cmd);
	void visit_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::Block<AST::Cmd>>& macro_body);
	void visit_regular_command(const IR::RegularCommand& regular_cmd);

	std::shared_ptr<StackNode> stack;

private:
	friend struct IR::MacroCall;

	//settings
	bool stop_on_fail;
	bool assume_yes;
	std::string invalidate;
	bool dry;

	std::vector<std::shared_ptr<IR::TestRun>> tests_runs;
	std::vector<std::shared_ptr<IR::Test>> up_to_date_tests;

	void delete_snapshot_with_children(const std::shared_ptr<IR::Test>& test);
	void invalidate_tests();
	void check_cache_missed_tests();
	void get_up_to_date_tests();
	std::shared_ptr<IR::TestRun> add_test_to_plan(const std::shared_ptr<IR::Test>& test);
	void build_test_plan();
	void init();

	std::shared_ptr<IR::Controller> current_controller;
	std::shared_ptr<IR::Test> current_test;
	Reporter reporter;

	void restore_parents_controllers_if_needed(const std::shared_ptr<IR::Test>& test);
	void create_networks_if_needed(const std::shared_ptr<IR::Test>& test);
	void install_new_controllers_if_needed(const std::shared_ptr<IR::Test>& test);
	void resume_parents_vms(const std::shared_ptr<IR::Test>& test);
	void suspend_all_vms(const std::shared_ptr<IR::Test>& test);
	void create_all_controllers_snapshots(const std::shared_ptr<IR::Test>& test);

	void stop_all_vms(const std::shared_ptr<IR::Test>& test);
};
