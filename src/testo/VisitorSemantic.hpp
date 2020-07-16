
#pragma once

#include "IR/Test.hpp"
#include "IR/Macro.hpp"
#include "IR/Action.hpp"
#include "IR/Expr.hpp"
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
	void visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates);
	void visit_select_text(const IR::SelectText& text);
	void visit_select_js(const IR::SelectJS& js);
	void visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable);
	void visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click);
	void visit_mouse_hold(const IR::MouseHold& mouse_hold);
	void visit_mouse_release(const IR::MouseRelease& mouse_release);
	void visit_mouse(const IR::Mouse& mouse);
	void visit_abort(const IR::Abort& abort);
	void visit_print(const IR::Print& print);
	void visit_type(const IR::Type& type);
	void visit_press(const IR::Press& press);
	void visit_key_combination(std::shared_ptr<AST::KeyCombination> combination);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec);
	void visit_hold(const IR::Hold& hold);
	void visit_release(const IR::Release& release);
	void visit_plug(const IR::Plug& plug);
	void visit_start(const IR::Start& start);
	void visit_stop(const IR::Stop& stop);
	void visit_shutdown(const IR::Shutdown& shutdown);
	void visit_exec(const IR::Exec& exec);
	void visit_copy(const IR::Copy& copy);
	void visit_wait(const IR::Wait& wait);
	void visit_sleep(const IR::Sleep& sleep);
	void visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	std::vector<std::string> visit_range(const IR::Range& range);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);
	void visit_cycle_control(const IR::CycleControl& cycle_control);

	void visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr);
	void visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable);
	void visit_detect_parented(std::shared_ptr<AST::SelectParentedExpr> parented);
	void visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop);

	std::optional<bool> visit_expr(std::shared_ptr<AST::IExpr> expr);
	std::optional<bool> visit_binop(std::shared_ptr<AST::BinOp> binop);
	bool visit_defined(const IR::Defined& defined);
	bool visit_comparison(const IR::Comparison& defined);
	std::optional<bool> visit_factor(std::shared_ptr<AST::IFactor> factor);
	std::optional<bool> visit_check(const IR::Check& check);

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
