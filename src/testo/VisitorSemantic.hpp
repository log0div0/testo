
#pragma once

#include "IR/Test.hpp"
#include "IR/Macro.hpp"
#include "IR/Action.hpp"
#include "TemplateLiterals.hpp"

#include <set>
#include <unordered_set>

struct VisitorSemantic {
	VisitorSemantic(const nlohmann::json& config);

	void visit();
	void visit_macro(std::shared_ptr<IR::Macro> macro);
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers);
	void visit_mouse_move_selectable(std::shared_ptr<AST::MouseSelectable> mouse_selectable);
	void visit_mouse_move_click(std::shared_ptr<AST::MouseMoveClick> mouse_move_click);
	void visit_mouse(std::shared_ptr<AST::Mouse> mouse);
	void visit_abort(const IR::Abort& abort);
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

	void visit_machine(std::shared_ptr<IR::Machine> machine);
	void visit_flash(std::shared_ptr<IR::FlashDrive> flash); //flash drive
	void visit_network(std::shared_ptr<IR::Network> network); //flash drive
	nlohmann::json visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx);
	void visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx);

	void validate_js(const std::string& script);

	bool is_button(const Token& t) const;

	std::set<std::string> keys;

	std::string prefix;

	//bool is for "requires a name"
	using attr_ctx = std::unordered_map<std::string, std::pair<bool, Token::category>>;
	std::unordered_map<std::string, attr_ctx> attr_ctxs;
	template_literals::Parser template_parser;
	std::shared_ptr<StackNode> stack;

	std::unordered_set<std::shared_ptr<IR::Macro>> visited_macros;
	std::unordered_set<std::shared_ptr<IR::Machine>> visited_machines;
	std::unordered_set<std::shared_ptr<IR::FlashDrive>> visited_flash_drives;
	std::unordered_set<std::shared_ptr<IR::Network>> visited_networks;

	std::shared_ptr<IR::Test> current_test;
};
