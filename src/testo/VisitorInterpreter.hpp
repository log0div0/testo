
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

struct VisitorInterpreter {
	struct InterpreterException: public std::exception {
			explicit InterpreterException():
				std::exception()
			{
				msg = "";
			}

			const char* what() const noexcept override {
				return msg.c_str();
			}
		protected:
			std::string msg;
	};

	struct ActionException: public InterpreterException {
		explicit ActionException(std::shared_ptr<AST::Node> node, std::shared_ptr<IR::Machine> vmc):
			InterpreterException(), node(node), vmc(vmc)
		{
			msg = std::string(node->begin()) + ": Error while performing action " + std::string(*node) + " ";
			if (vmc) {
				msg += "on virtual machine ";
				msg += vmc->name();
			}
		}
	private:
		std::shared_ptr<AST::Node> node;
		std::shared_ptr<IR::Machine> vmc;
	};

	struct AbortException: public InterpreterException {
		explicit AbortException(std::shared_ptr<AST::Abort> node, std::shared_ptr<IR::Machine> vmc, const std::string& message):
			InterpreterException(), node(node), vmc(vmc)
		{
			msg = std::string(node->begin()) + ": Caught abort action ";
			if (vmc) {
				msg += "on virtual machine ";
				msg += vmc->name();
			}

			msg += " with message: ";
			msg += message;
		}
	private:
		std::shared_ptr<AST::Node> node;
		std::shared_ptr<IR::Machine> vmc;
	};


	struct CycleControlException: public InterpreterException {
		explicit CycleControlException(const Token& token):
			InterpreterException(), token(token)
		{
			msg = std::string(token.begin()) + " error: cycle control action has not a correcponding cycle";
		}

		Token token;
	};

	VisitorInterpreter(const nlohmann::json& config);

	void visit();
	void visit_test(std::shared_ptr<IR::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_abort(std::shared_ptr<AST::Abort> abort);
	void visit_print(std::shared_ptr<AST::Print> print_action);
	void visit_type(const IR::Type& type);
	void visit_wait(const IR::Wait& wait);
	void visit_sleep(std::shared_ptr<AST::Sleep> sleep);
	nn::Tensor visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Tensor& input);
	nn::Point visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Tensor& input);
	nn::Point visit_mouse_specifier_default_centering(const nn::Tensor& input);
	nn::Point visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Point& input);
	nn::Point visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers, const nn::Tensor& input);
	nn::Point visit_select_js(std::shared_ptr<AST::Selectable<AST::SelectJS>> js, stb::Image& screenshot);
	bool visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image& screenshot);
	bool visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image& screenshot);
	bool visit_detect_unop(std::shared_ptr<AST::SelectUnOp> unop, stb::Image& screenshot);
	bool visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image& screenshot);
	void visit_press(const IR::Press& press);
	void visit_hold(std::shared_ptr<AST::Hold> hold);
	void visit_release(std::shared_ptr<AST::Release> release);
	void visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable);
	void visit_mouse(std::shared_ptr<AST::Mouse> mouse);
	void visit_mouse_move_click(std::shared_ptr<AST::MouseMoveClick> mouse_move_click);
	void visit_mouse_move_coordinates(std::shared_ptr<AST::MouseCoordinates> coordinates);
	void visit_mouse_hold(std::shared_ptr<AST::MouseHold> mouse_hold);
	void visit_mouse_release(std::shared_ptr<AST::MouseRelease> mouse_release);
	void visit_mouse_wheel(std::shared_ptr<AST::MouseWheel> mouse_wheel);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval);
	void visit_plug(std::shared_ptr<AST::Plug> plug);
	void visit_plug_nic(std::shared_ptr<AST::Plug> plug);
	void visit_plug_link(std::shared_ptr<AST::Plug> plug);
	void visit_plug_dvd(std::shared_ptr<AST::Plug> plug);
	void plug_flash(std::shared_ptr<AST::Plug> plug);
	void unplug_flash(std::shared_ptr<AST::Plug> plug);
	void visit_start(std::shared_ptr<AST::Start> start);
	void visit_stop(std::shared_ptr<AST::Stop> stop);
	void visit_shutdown(std::shared_ptr<AST::Shutdown> shutdown);
	void visit_exec(const IR::Exec& exec);
	void visit_copy(const IR::Copy& copy);
	void visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);

	bool visit_expr(std::shared_ptr<AST::IExpr> expr);
	bool visit_binop(std::shared_ptr<AST::BinOp> binop);
	bool visit_factor(std::shared_ptr<AST::IFactor> factor);
	bool visit_comparison(std::shared_ptr<AST::Comparison> comparison);
	bool visit_defined(const IR::Defined& defined);
	bool visit_check(const IR::Check& check);

	js::Value eval_js(const std::string& script, stb::Image& screenshot);

	std::string test_cksum(std::shared_ptr<IR::Test> test) const;
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
};
