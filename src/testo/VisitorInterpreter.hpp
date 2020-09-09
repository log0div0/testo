
#pragma once
#include "coro/Timer.h"
#include "IR/Test.hpp"
#include "IR/Action.hpp"
#include "IR/Expr.hpp"
#include "TemplateLiterals.hpp"
#include "Reporter.hpp"
#include "js/Context.hpp"
#include <nn/OCR.hpp>
#include <vector>
#include <list>
#include <unordered_map>

struct VisitorInterpreter {
	VisitorInterpreter(const nlohmann::json& config);

	void visit();
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_abort(const IR::Abort& abort);
	void visit_print(const IR::Print& print);
	void visit_type(const IR::Type& type);
	void visit_wait(const IR::Wait& wait);
	void visit_sleep(const IR::Sleep& sleep);
	nn::Tensor visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Tensor& input);
	nn::Point visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Tensor& input);
	nn::Point visit_mouse_specifier_default_centering(const nn::Tensor& input);
	nn::Point visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Point& input);
	nn::Point visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers, const nn::Tensor& input);
	nn::Tensor visit_select_text(const IR::SelectText& text, stb::Image& screenshot);
	bool visit_detect_js(const IR::SelectJS& js, stb::Image& screenshot);
	nn::Point visit_select_js(const IR::SelectJS& js, stb::Image& screenshot);
	bool visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image& screenshot);
	bool visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image& screenshot);
	bool visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image& screenshot);
	void visit_press(const IR::Press& press);
	void visit_hold(const IR::Hold& hold);
	void visit_release(const IR::Release& release);
	void visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable);
	void visit_mouse(const IR::Mouse& mouse);
	void visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click);
	void visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates);
	void visit_mouse_hold(const IR::MouseHold& mouse_hold);
	void visit_mouse_release(const IR::MouseRelease& mouse_release);
	void visit_mouse_wheel(std::shared_ptr<AST::MouseWheel> mouse_wheel);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval);
	void visit_plug(const IR::Plug& plug);
	void visit_plug_nic(const IR::Plug& plug);
	void visit_plug_link(const IR::Plug& plug);
	void visit_plug_dvd(const IR::Plug& plug);
	void visit_plug_flash(const IR::Plug& plug);
	void visit_unplug_flash(const IR::Plug& plug);
	void visit_start(const IR::Start& start);
	void visit_stop(const IR::Stop& stop);
	void visit_shutdown(const IR::Shutdown& shutdown);
	void visit_exec(const IR::Exec& exec);
	void visit_copy(const IR::Copy& copy);
	void visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);
	std::vector<std::string> visit_range(const IR::Range& range);

	bool visit_expr(std::shared_ptr<AST::IExpr> expr);
	bool visit_binop(std::shared_ptr<AST::BinOp> binop);
	bool visit_factor(std::shared_ptr<AST::IFactor> factor);
	bool visit_comparison(const IR::Comparison& comparison);
	bool visit_defined(const IR::Defined& defined);
	bool visit_check(const IR::Check& check);

	js::Value eval_js(const std::string& script, stb::Image& screenshot);

	template_literals::Parser template_parser;

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

	coro::Timer timer;

	std::unordered_map<std::string, std::vector<std::string>> charmap;

	std::shared_ptr<js::Context> js_current_ctx;

	std::shared_ptr<IR::Machine> vmc;

	Reporter reporter;
};
