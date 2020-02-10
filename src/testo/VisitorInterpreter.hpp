
#pragma once
#include "coro/Timer.h"
#include "Node.hpp"
#include "Register.hpp"
#include "TemplateParser.hpp"
#include "Reporter.hpp"
#include "quickjs/Runtime.hpp"
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
		explicit ActionException(std::shared_ptr<AST::Node> node, std::shared_ptr<VmController> vmc):
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
		std::shared_ptr<VmController> vmc;
	};

	struct AbortException: public InterpreterException {
		explicit AbortException(std::shared_ptr<AST::Abort> node, std::shared_ptr<VmController> vmc, const std::string& message):
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
		std::shared_ptr<VmController> vmc;
	};


	struct CycleControlException: public InterpreterException {
		explicit CycleControlException(const Token& token):
			InterpreterException(), token(token)
		{
			msg = std::string(token.pos()) + " error: cycle control action has not a correcponding cycle";
		}

		Token token;
	};

	VisitorInterpreter(Register& reg, const nlohmann::json& config);

	void visit(std::shared_ptr<AST::Program> program);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IAction> action);
	void visit_abort(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Abort> abort);
	void visit_print(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Print> print_action);
	void visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Type> type);
	void visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Wait> wait);
	bool visit_select_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image& screenshot);
	bool visit_select_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image& screenshot);
	bool visit_select_unop(std::shared_ptr<AST::SelectUnOp> unop, stb::Image& screenshot);
	bool visit_select_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image& screenshot);
	void visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Press> press);
	void visit_mouse_event(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseEvent> mouse_event);
	void visit_key_spec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::KeySpec> key_spec);
	void visit_plug(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void visit_plug_nic(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void visit_plug_link(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void visit_plug_dvd(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void plug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void unplug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug);
	void visit_start(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Start> start);
	void visit_stop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Stop> stop);
	void visit_shutdown(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Shutdown> shutdown);
	void visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Exec> exec);
	void visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Copy> copy);
	void visit_macro_call(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IfClause> if_clause);
	void visit_for_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ForClause> for_clause);

	bool visit_expr(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IExpr> expr);
	bool visit_binop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::BinOp> binop);
	bool visit_factor(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IFactor> factor);
	bool visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Comparison> comparison);
	bool visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Check> check);

	bool eval_js(const std::string& script, stb::Image& screenshot);

	std::string test_cksum(std::shared_ptr<AST::Test> test) const;

	Register& reg;
	template_literals::Parser template_parser;

private:
	//settings
	bool stop_on_fail;
	bool assume_yes;
	std::string test_spec, exclude, invalidate;

	std::list<std::shared_ptr<AST::Test>> tests_to_run;
	std::vector<std::shared_ptr<AST::Test>> up_to_date_tests;
	std::vector<std::shared_ptr<AST::Test>> cache_missed_tests;

	void setup_vars(std::shared_ptr<AST::Program> program);
	void reset_cache();

	bool parent_is_ok(std::shared_ptr<AST::Test> test, std::shared_ptr<AST::Test> parent,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator end);

	void build_test_plan(std::shared_ptr<AST::Test> test,
		std::list<std::shared_ptr<AST::Test>>& test_plan,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator end);

	bool is_cached(std::shared_ptr<AST::Test> test) const;
	bool is_cache_miss(std::shared_ptr<AST::Test> test) const;
	void check_up_to_date_tests(std::list<std::shared_ptr<AST::Test>>& tests_queue);
	void resolve_tests(const std::list<std::shared_ptr<AST::Test>>& tests_queue);
	void update_progress();

	void stop_all_vms(std::shared_ptr<AST::Test> test) {
		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->is_defined()) {
				if (vmc->vm->state() != VmState::Stopped) {
					vmc->vm->stop();
				}
				vmc->set_metadata("current_state", "");
			}
		}
	}
	coro::Timer timer;

	std::vector<std::shared_ptr<AST::Controller>> flash_drives;

	std::unordered_map<std::string, std::vector<std::string>> charmap;

	std::string license_status;
	quickjs::Runtime js_runtime;
};
