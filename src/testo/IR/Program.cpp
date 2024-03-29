
#include "Program.hpp"

#include "../visitors/VisitorSemantic.hpp"
#include "../visitors/VisitorInterpreter.hpp"
#include "../backends/Environment.hpp"
#include "../resolver/Resolver.hpp"
#include "../parser/Parser.hpp"
#include "../Exceptions.hpp"
#include "../Logger.hpp"

#include <wildcards.hpp>

namespace IR {

Program::Program(const std::shared_ptr<AST::Program>& ast, const ProgramConfig& config_): config(config_) {
	TRACE();

	if (program != nullptr) {
		throw std::runtime_error("Only one instance of IR::Program can exists");
	}
	program = this;

	setup_stack();
	collect_top_level_objects(ast);
	validate_special_params();
	setup_tests_parents();
}

Program::~Program() {
	TRACE();

	program = nullptr;
}

void Program::validate() {
	TRACE();
	env->setup(config);
	VisitorSemantic semantic(config);
	semantic.visit();
}

void Program::run() {
	TRACE();
	VisitorInterpreter runner(config);
	runner.visit();
}

std::shared_ptr<Macro> Program::get_macro_or_throw(const std::string& name) {
	return get_or_throw(name, macros);
}

std::shared_ptr<Param> Program::get_param_or_throw(const std::string& name) {
	return get_or_throw(name, params);
}

std::shared_ptr<Machine> Program::get_machine_or_throw(const std::string& name) {
	return get_or_throw(name, machines);
}

std::shared_ptr<FlashDrive> Program::get_flash_drive_or_throw(const std::string& name) {
	return get_or_throw(name, flash_drives);
}

std::shared_ptr<Network> Program::get_network_or_throw(const std::string& name) {
	return get_or_throw(name, networks);
}


std::shared_ptr<Test> Program::get_test_or_null(const std::string& name) {
	for (auto& test: ordered_tests) {
		if (test->name() == name) {
			return test;
		}
	}
	return nullptr;
}

std::shared_ptr<Macro> Program::get_macro_or_null(const std::string& name) {
	return get_or_null(name, macros);
}

std::shared_ptr<Param> Program::get_param_or_null(const std::string& name) {
	return get_or_null(name, params);
}

std::shared_ptr<Machine> Program::get_machine_or_null(const std::string& name) {
	return get_or_null(name, machines);
}

std::shared_ptr<FlashDrive> Program::get_flash_drive_or_null(const std::string& name) {
	return get_or_null(name, flash_drives);
}

std::shared_ptr<Network> Program::get_network_or_null(const std::string& name) {
	return get_or_null(name, networks);
}

std::map<std::string, std::string> testo_default_params = {
	{"TESTO_WAIT_DEFAULT_TIMEOUT", "1m"},
	{"TESTO_WAIT_DEFAULT_INTERVAL", "1s"},
	{"TESTO_CHECK_DEFAULT_TIMEOUT", "1ms"},
	{"TESTO_CHECK_DEFAULT_INTERVAL", "1s"},
	{"TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT", "1m"},
	{"TESTO_PRESS_DEFAULT_INTERVAL", "30ms"},
	{"TESTO_TYPE_DEFAULT_INTERVAL", "30ms"},
	{"TESTO_EXEC_DEFAULT_TIMEOUT", "10m"},
	{"TESTO_COPY_DEFAULT_TIMEOUT", "10m"},
	{"TESTO_SHUTDOWN_DEFAULT_TIMEOUT", "1m"},
#ifdef __aarch64__
	{"TESTO_DISK_DEFAULT_BUS", "scsi"},
#else
	{"TESTO_DISK_DEFAULT_BUS", "ide"},
#endif
	{"TESTO_SNAPSHOT_DEFAULT_POLICY", "always"},
};

std::vector<std::string> testo_timeout_params = {
	"TESTO_WAIT_DEFAULT_TIMEOUT",
	"TESTO_WAIT_DEFAULT_INTERVAL",
	"TESTO_CHECK_DEFAULT_TIMEOUT",
	"TESTO_CHECK_DEFAULT_INTERVAL",
	"TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT",
	"TESTO_PRESS_DEFAULT_INTERVAL",
	"TESTO_TYPE_DEFAULT_INTERVAL",
	"TESTO_EXEC_DEFAULT_TIMEOUT",
	"TESTO_COPY_DEFAULT_TIMEOUT",
	"TESTO_SHUTDOWN_DEFAULT_TIMEOUT",
};

void Program::setup_stack() {
	TRACE();
	auto predefined = std::make_shared<StackNode>();
	predefined->params = testo_default_params;
	stack = std::make_shared<StackNode>();
	stack->parent = predefined;
	for (size_t i = 0; i < config.params_names.size(); ++i) {
		stack->params[config.params_names.at(i)] = config.params_values.at(i);
	}
}

void Program::collect_top_level_objects(const std::shared_ptr<AST::Program>& ast) {
	TRACE();
	for (auto stmt: ast->stmts) {
		visit_stmt(stmt);
	}
}

void Program::visit_stmt(const std::shared_ptr<AST::Stmt>& stmt) {
	if (auto p = std::dynamic_pointer_cast<AST::Test>(stmt)) {
		collect_test(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Stmt>>(stmt)) {
		visit_macro_call({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Macro>(stmt)) {
		collect_macro(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Param>(stmt)) {
		collect_param(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Controller>(stmt)) {
		if (p->controller.type() == Token::category::machine) {
			collect_machine(p);
		} else if (p->controller.type() == Token::category::flash) {
			collect_flash_drive(p);
		} else if (p->controller.type() == Token::category::network) {
			collect_network(p);
		} else {
			throw std::runtime_error("Unknown controller type");
		}
	} else {
		throw std::runtime_error("Unknown statement");
	}
}

void Program::visit_statement_block(const std::shared_ptr<AST::Block<AST::Stmt>>& stmt_block) {
	for (auto stmt: stmt_block->items) {
		if (auto p = std::dynamic_pointer_cast<AST::Macro>(stmt)) {
			throw ExceptionWithPos(stmt->begin(), "Error: nested macro declarations are not supported");
		} else if (auto p = std::dynamic_pointer_cast<AST::Param>(stmt)) {
			throw ExceptionWithPos(stmt->begin(), "Error: param declaration inside macros is not supported");
		}

		visit_stmt(stmt);
	}
}

void Program::collect_test(const std::shared_ptr<AST::Test>& test) {
	auto inserted = insert_object(test, tests);
	ordered_tests.push_back(inserted);
}

void Program::visit_macro(std::shared_ptr<IR::Macro> macro) {
	auto result = visited_macros.insert(macro);
	if (!result.second) {
		return;
	}

	macro->validate();
}

void Program::visit_macro_call(const IR::MacroCall& macro_call) {
	current_macro_call_stack.push_back(macro_call.ast_node);
	macro_call.visit_semantic<AST::Stmt>(this);
	current_macro_call_stack.pop_back();
}

void Program::visit_macro_body(const std::shared_ptr<AST::Block<AST::Stmt>>& macro_body) {
	visit_statement_block(macro_body);
}

void Program::collect_macro(const std::shared_ptr<AST::Macro>& macro) {
	insert_object(macro, macros);
}

void Program::collect_param(const std::shared_ptr<AST::Param>& param_ast) {
	auto param = insert_object(param_ast, params);
	if (stack->params.count(param->name())) {
		throw ExceptionWithPos(param_ast->begin(), "Error: param \"" + param->name()
			+ "\" is already defined as a command line argument");
	}
	stack->params[param->name()] = param->value();
}
void Program::collect_machine(const std::shared_ptr<AST::Controller>& machine) {
	insert_controller(machine, machines);
}
void Program::collect_flash_drive(const std::shared_ptr<AST::Controller>& flash) {
	insert_controller(flash, flash_drives);
}
void Program::collect_network(const std::shared_ptr<AST::Controller>& network) {
	insert_controller(network, networks);
}

bool check_if_time_interval(const std::string& time) {
	std::string number;

	size_t i = 0;

	for (; i < time.length(); ++i) {
		if (isdigit(time[i])) {
			number += time[i];
		} else {
			break;
		}
	}

	if (!number.length()) {
		return false;
	}

	if (time[i] == 's' || time[i] == 'h') {
		return (i == time.length() - 1);
	}

	if (time[i] == 'm') {
		if (i == time.length() - 1) {
			return true;
		}

		if (time.length() > i + 2) {
			return false;
		}
		return time[i + 1] == 's';
	}

	return false;

}

void Program::validate_special_params() {
	TRACE();
	for (auto& param: testo_timeout_params) {
		std::string value = resolve_top_level_param(param);
		if (!check_if_time_interval(value)) {
			throw std::runtime_error("Can't convert parameter " + param + " value \"" + value + "\" to time interval");
		}
	}
}

void Program::setup_tests_parents() {
	TRACE();
	for (auto& test: ordered_tests) {
		auto test_name = test->name();

		if (config.validate_test_name(test_name)) {
			setup_test_parents(test);
		}
	}
}

void Program::setup_test_parents(const std::shared_ptr<Test>& test) {
	for (auto& t: all_selected_tests) {
		if (t == test) {
			return;
		}
	}

	all_selected_tests.push_back(test);

	auto parent_names = test->parent_names();

	for (size_t i = 0; i < parent_names.size(); i++) {
		auto parent_name = parent_names[i];

		if (parent_name == test->name()) {
			throw ExceptionWithPos(test->ast_node->parents->at(i)->begin(), "Error: can't specify test as a parent to itself " + parent_name);
		}

		auto it = tests.find(parent_name);
		if (it == tests.end()) {
			throw ExceptionWithPos(test->ast_node->parents->at(i)->begin(), "Error: unknown test: " + parent_name);
		}

		auto parent = it->second;

		if (std::find(test->parents.begin(), test->parents.end(), parent) == test->parents.end()) {
			test->parents.push_back(parent);
		} else {
			throw ExceptionWithPos(test->ast_node->parents->at(i)->begin(), "Error: this test was already specified in parent list " + parent_name);
		}

		if (std::find_if(parent->children.begin(), parent->children.end(), [&](const auto& child) { return child.lock() == test; }) == parent->children.end()) {
			parent->children.push_back(test);
		}

		setup_test_parents(parent);
	}

	for (auto& dep: test->depends_on()) {
		std::shared_ptr<Test> dep_test = get_test_or_null(dep);
		if (!dep_test) {
			throw std::runtime_error("Test " + test->name() + " depends on unknown test " + dep);
		}

		setup_test_parents(dep_test);
	}
}

std::string Program::resolve_top_level_param(const std::string& name) const {
	std::string param = stack->find_param(name);
	return Resolver(param).resolve(stack, nullptr);
}

Program* program = nullptr;

}
