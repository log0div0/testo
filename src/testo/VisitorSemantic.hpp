
#pragma once

#include "Node.hpp"
#include "Register.hpp"
#include "TemplateParser.hpp"
#include "backends/Environment.hpp"

#include <set>

struct VisitorSemantic {
	VisitorSemantic(std::shared_ptr<Register> reg, const nlohmann::json& config);
	void setup_macros(std::shared_ptr<AST::Test> test);
	void setup_macros_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void setup_macros_action(std::shared_ptr<AST::IAction> action);
	void setup_macros_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void setup_macros_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void setup_macros_for_clause(std::shared_ptr<AST::ForClause> for_clause);
	void setup_test(std::shared_ptr<AST::Test> test);
	void setup_tests(std::shared_ptr<AST::Program> program);
	void setup_vars(std::shared_ptr<AST::Program> program);

	void visit(std::shared_ptr<AST::Program> program);
	void visit_stmt(std::shared_ptr<AST::IStmt> stmt);
	void visit_macro(std::shared_ptr<AST::Macro> macro);
	void visit_param(std::shared_ptr<AST::Param> param);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers);
	void visit_mouse_move_selectable(std::shared_ptr<AST::MouseSelectable> mouse_selectable);
	void visit_mouse_move_click(std::shared_ptr<AST::MouseMoveClick> mouse_move_click);
	void visit_mouse(std::shared_ptr<AST::Mouse> mouse);
	void visit_press(std::shared_ptr<AST::Press> press);
	void visit_key_combination(std::shared_ptr<AST::KeyCombination> combination);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec);
	void visit_plug(std::shared_ptr<AST::Plug> plug);
	void visit_exec(std::shared_ptr<AST::Exec> exec);
	void visit_wait(std::shared_ptr<AST::Wait> wait);
	void visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void visit_range(std::shared_ptr<AST::Range> range);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);

	void visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr);
	void visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable);
	void visit_detect_unop(std::shared_ptr<AST::SelectUnOp> unop);
	void visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop);

	void visit_expr(std::shared_ptr<AST::IExpr> expr);
	void visit_binop(std::shared_ptr<AST::BinOp> binop);
	void visit_factor(std::shared_ptr<AST::IFactor> factor);
	void visit_check(std::shared_ptr<AST::Check> check);

	void visit_controller(std::shared_ptr<AST::Controller> controller);
	void visit_machine(std::shared_ptr<AST::Controller> machine);
	void visit_flash(std::shared_ptr<AST::Controller> flash); //flash drive
	void visit_network(std::shared_ptr<AST::Controller> network); //flash drive
	nlohmann::json visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx);
	void visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx);

	void validate_js(const std::string& script);

	bool is_button(const Token& t) const;

	std::set<std::string> keys;

	std::string prefix;
	std::string test_spec, exclude;

	std::shared_ptr<Register> reg;

	//bool is for "requires a name"
	using attr_ctx = std::unordered_map<std::string, std::pair<bool, Token::category>>;
	std::unordered_map<std::string, attr_ctx> attr_ctxs;
	template_literals::Parser template_parser;

	std::set<std::string> testo_timeout_params;

	std::list<std::shared_ptr<AST::Test>> tests_queue;
};
