
#pragma once

#include "Node.hpp"
#include "Register.hpp"
#include "StinkingPileOfShit.hpp"
#include <vector>
#include <list>

struct VisitorInterpreter {
	struct StackEntry {
		StackEntry(bool is_terminate): is_terminate(is_terminate) {}

		void define(const std::string& name, const std::string& value) {
			vars[name] = value;
		}


		bool is_defined(const std::string& name) {
			return (vars.find(name) != vars.end());
		}

		std::string ref(const std::string& name) {
			auto found = vars.find(name);

			if (found != vars.end()) {
				return found->second;
			} else {
				throw std::runtime_error(std::string("Var ") + name + " not defined");
			}
		}

		bool is_terminate;
		std::unordered_map<std::string, std::string> vars;
	};

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
				msg += vmc->vm->name();
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
	void visit_controller(std::shared_ptr<AST::Controller> controller);
	void visit_flash(std::shared_ptr<AST::Controller> flash);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IAction> action);
	void visit_abort(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Abort> abort);
	void visit_print(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Print> print_action);
	void visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Type> type);
	void visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Wait> wait);
	void visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Press> press);
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
	std::string resolve_var(std::shared_ptr<VmController> vmc, const std::string& var);
	std::string visit_word(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Word> word);
	bool visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Comparison> comparison);
	bool visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Check> check);

	std::string test_cksum(std::shared_ptr<AST::Test> test);

	Register& reg;

private:
	//settings
	bool stop_on_fail;
	std::string test_spec, exclude, invalidate;

	std::vector<StackEntry> local_vars;

	void do_print() {}

	template <typename First, typename... Rest>
	void do_print(First first, Rest... rest) {
		std::cout << first;
		do_print(std::forward<Rest>(rest)...);
	}

	template <typename... Args>
	void print(Args... args) {
		std::cout << "[";
		std::cout << std::setw(3);
		std::cout << std::round(current_progress);
		std::cout << std::setw(0);
		std::cout << '%' << "] ";
		do_print(std::forward<Args>(args)...);
		std::cout << std::endl;
	}

	std::vector<std::shared_ptr<AST::Test>> succeeded_tests;
	std::vector<std::shared_ptr<AST::Test>> failed_tests;
	std::vector<std::shared_ptr<AST::Test>> up_to_date_tests;

	void print_statistics() const;

	void setup_vars(std::shared_ptr<AST::Program> program);
	void reset_cache();


	bool parent_is_ok(std::shared_ptr<AST::Test> test, std::shared_ptr<AST::Test> parent,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator end);

	void build_test_plan(std::shared_ptr<AST::Test> test,
		std::list<std::shared_ptr<AST::Test>>& test_plan,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
		std::list<std::shared_ptr<AST::Test>>::reverse_iterator end);

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

	float current_progress = 0;
	float progress_step = 0;

	std::chrono::system_clock::time_point start_timestamp;

	std::list<std::shared_ptr<AST::Test>> tests_to_run;
	std::vector<std::shared_ptr<AST::Controller>> flash_drives;

	StinkingPileOfShit shit;
	std::unordered_map<char, std::vector<std::string>> charmap;

};
