
#pragma once

#include "IR/Test.hpp"
#include "IR/Macro.hpp"
#include "IR/Action.hpp"
#include "IR/Command.hpp"
#include "IR/Expr.hpp"
#include "TemplateLiterals.hpp"

#include <set>
#include <unordered_set>
#include <unordered_map>

enum class Tribool: uint8_t { no, yes, maybe };

inline Tribool operator^(bool a, Tribool b) {
	if (a) {
		switch (b) {
			case Tribool::yes:
				return Tribool::no;
			case Tribool::no:
				return Tribool::yes;
			default:
				return Tribool::maybe;
		}
	} else {
		return b;
	}
}

struct VisitorSemanticConfig {
	std::string prefix;

	void validate() const;
};

struct VisitorSemantic {
	VisitorSemantic(const VisitorSemanticConfig& config);

	void visit();
	void visit_macro(std::shared_ptr<IR::Macro> macro);
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::ICmd> cmd);
	void visit_regular_command(const IR::RegularCommand& regular_cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_action_vm(std::shared_ptr<AST::IAction> action);
	void visit_action_fd(std::shared_ptr<AST::IAction> action);
	void visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers);
	void visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates);
	void visit_select_text(const IR::SelectText& text);
	void visit_select_js(const IR::SelectJS& js);
	void visit_select_img(const IR::SelectImg& img);
	void visit_select_homm3(const IR::SelectHomm3& homm3);
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
	void visit_key_spec(const IR::KeySpec& key_spec);
	void visit_hold(const IR::Hold& hold);
	void visit_release(const IR::Release& release);
	void visit_plug(const IR::Plug& plug);
	void visit_plug_flash(const IR::PlugFlash& plug_flash);
	void visit_plug_dvd(const IR::PlugDVD& plug_dvd, bool is_on);
	void visit_plug_nic(const IR::PlugNIC& plug_nic);
	void visit_plug_link(const IR::PlugLink& plug_link);
	void visit_plug_hostdev(const IR::PlugHostDev& plug_hostdev);
	void visit_start(const IR::Start& start);
	void visit_stop(const IR::Stop& stop);
	void visit_shutdown(const IR::Shutdown& shutdown);
	void visit_exec(const IR::Exec& exec);
	void visit_copy(const IR::Copy& copy);
	void visit_wait(const IR::Wait& wait);
	void visit_sleep(const IR::Sleep& sleep);
	void visit_cmd_macro_call(const IR::MacroCall& macro_call);
	void visit_action_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::MacroBodyCommand>& macro_body);
	void visit_macro_body(const std::shared_ptr<AST::MacroBodyAction>& macro_body);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	std::vector<std::string> visit_range(const IR::Range& range);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);
	void visit_cycle_control(const IR::CycleControl& cycle_control);

	void visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr);
	void visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable);
	void visit_detect_parented(std::shared_ptr<AST::SelectParentedExpr> parented);
	void visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop);

	Tribool visit_expr(std::shared_ptr<AST::IExpr> expr);
	Tribool visit_binop(std::shared_ptr<AST::BinOp> binop);
	Tribool visit_defined(const IR::Defined& defined);
	Tribool visit_comparison(const IR::Comparison& defined);
	Tribool visit_factor(std::shared_ptr<AST::IFactor> factor);
	Tribool visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented);
	Tribool visit_check(const IR::Check& check);

	void visit_machine(std::shared_ptr<IR::Machine> machine);
	void visit_flash(std::shared_ptr<IR::FlashDrive> flash); //flash drive
	void visit_network(std::shared_ptr<IR::Network> network); //flash drive
	nlohmann::json visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx);
	void visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx);

	void validate_js(const std::string& script);

	bool is_button(const Token& t) const;

	std::set<std::string> keys;

	std::string prefix;

	template_literals::Parser template_parser;
	std::shared_ptr<StackNode> stack;

	std::shared_ptr<IR::Controller> current_controller;

	std::unordered_set<std::shared_ptr<IR::Macro>> visited_macros;
	std::unordered_set<std::shared_ptr<IR::Machine>> visited_machines;
	std::unordered_set<std::shared_ptr<IR::FlashDrive>> visited_flash_drives;
	std::unordered_set<std::shared_ptr<IR::Network>> visited_networks;

	using attr_ctx = std::unordered_map<std::string, std::pair<bool, std::vector<Token::category>>>;
	std::unordered_map<std::string, attr_ctx> attr_ctxs;

	std::shared_ptr<IR::Test> current_test;
};
