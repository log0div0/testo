
#pragma once

#include "Node.hpp"
#include "Register.hpp"
#include <stack>

struct VisitorInterpreter {
	struct Stack {
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

		std::unordered_map<std::string, std::string> vars;
	};

	struct InterpreterException: public std::runtime_error {
		explicit InterpreterException(std::shared_ptr<AST::Node> node, std::shared_ptr<VmController> vm):
			std::runtime_error(""), node(node), vm(vm)
		{
			msg = std::string(node->begin()) + ": Error while performing action " + std::string(*node) + " ";
			if (vm) {
				msg += "on vm ";
				msg += vm->name();
			}
		}

		const char* what() const noexcept override {
			return msg.c_str();
		}

	private:
		std::string msg;
		std::shared_ptr<AST::Node> node;
		std::shared_ptr<VmController> vm;
	};

	VisitorInterpreter(Register& reg):
		reg(reg) {}

	void visit(std::shared_ptr<AST::Program> program);
	void visit_stmt(std::shared_ptr<AST::IStmt> stmt);
	void visit_controller(std::shared_ptr<AST::Controller> controller);
	void visit_flash(std::shared_ptr<AST::Controller> flash);
	void visit_snapshot(std::shared_ptr<AST::Snapshot> snapshot);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_vm_state(std::shared_ptr<AST::VmState> vm_state);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<VmController> vm, std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IAction> action);
	void visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Type> type);
	void visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Wait> wait);
	void visit_press(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Press> press);
	void visit_key_spec(std::shared_ptr<VmController> vm, std::shared_ptr<AST::KeySpec> key_spec);
	void visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_plug_nic(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_plug_link(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_plug_dvd(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void plug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void unplug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_start(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Start> start);
	void visit_stop(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Stop> stop);
	void visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Exec> exec);
	void visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Set> set);
	void visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<AST::CopyTo> copyto);
	void visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<AST::MacroCall> macro_call);
	void visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IfClause> if_clause);

	bool visit_expr(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IExpr> expr);
	bool visit_binop(std::shared_ptr<VmController> vm, std::shared_ptr<AST::BinOp> binop);
	bool visit_factor(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IFactor> factor);
	std::string resolve_var(std::shared_ptr<VmController> vm, const std::string& var);
	std::string visit_word(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Word> word);
	bool visit_comparison(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Comparison> comparison);

	void apply_actions(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> snapshot, bool recursive = false);
	bool resolve_state(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> snapshot);
	bool check_config_relevance(nlohmann::json new_config, nlohmann::json old_config) const;
	std::string snapshot_cksum(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> snapshot);
	std::string cksum(std::shared_ptr<AST::Controller> controller);

	Register& reg;

private:
	std::stack<Stack> local_vars;

	template <typename... Args>
	void print(Args... args) {
		std::cout << "[";
		std::cout << std::setw(3);
		std::cout << current_progress;
		std::cout << std::setw(0);
		std::cout << "\%] ";
		(std::cout << ... << args);
		std::cout << std::endl;
	}

	void setup_progress_vars(std::shared_ptr<AST::Program> program);
	void update_progress();

	uint16_t current_progress = 0;
	uint16_t progress_step = 0;
	uint16_t original_remainder = 0;
	uint16_t current_remainder = 0;
	uint32_t tests_num = 0;
};
