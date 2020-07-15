
#include "Program.hpp"
#include "../VisitorSemantic.hpp"
#include "../VisitorInterpreter.hpp"
#include "../backends/Environment.hpp"
#include "nn/OnnxRuntime.hpp"
#include <wildcards.hpp>

namespace IR {

Program::Program(const std::shared_ptr<AST::Program>& ast, const nlohmann::json& config_): config(config_) {
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
	program = nullptr;
}

void Program::validate() {
	VisitorSemantic semantic(config);
	semantic.visit();
}

void Program::run() {
	env->setup();
	nn::OnnxRuntime onnx_runtime;
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


std::shared_ptr<Macro> Program::get_macro_or_null(const std::string& name) {
	return get_or_null(name, macros);
}

std::shared_ptr<Param> Program::get_param_or_null(const std::string& name) {
	return get_or_null(name, params);
}

std::shared_ptr<Machine> Program::get_machine_or_null(const std::string& name) {
	return get_or_throw(name, machines);
}

std::shared_ptr<FlashDrive> Program::get_flash_drive_or_null(const std::string& name) {
	return get_or_throw(name, flash_drives);
}

std::shared_ptr<Network> Program::get_network_or_null(const std::string& name) {
	return get_or_throw(name, networks);
}

std::map<std::string, std::string> testo_timeout_params = {
	{"TESTO_WAIT_DEFAULT_TIMEOUT", "1m"},
	{"TESTO_WAIT_DEFAULT_INTERVAL", "1s"},
	{"TESTO_CHECK_DEFAULT_TIMEOUT", "1ms"},
	{"TESTO_CHECK_DEFAULT_INTERVAL", "1s"},
	{"TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT", "1m"},
	{"TESTO_PRESS_DEFAULT_INTERVAL", "30ms"},
	{"TESTO_TYPE_DEFAULT_INTERVAL", "30ms"},
	{"TESTO_EXEC_DEFAULT_TIMEOUT", "10m"},
	{"TESTO_COPY_DEFAULT_TIMEOUT", "10m"},
};

void Program::setup_stack() {
	auto predefined = std::make_shared<StackNode>();
	predefined->vars = testo_timeout_params;
	stack = std::make_shared<StackNode>();
	stack->parent = predefined;
	for (auto& item: config.at("params")) {
		stack->vars[item.at("name").get<std::string>()] = item.at("value").get<std::string>();
	}
}

void Program::collect_top_level_objects(const std::shared_ptr<AST::Program>& ast) {
	for (auto stmt: ast->stmts) {
		if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
			collect_test(p->stmt);
		} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Macro>>(stmt)) {
			collect_macro(p->stmt);
		} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Param>>(stmt)) {
			collect_param(p->stmt);
		} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Controller>>(stmt)) {
			if (p->t.type() == Token::category::machine) {
				collect_machine(p->stmt);
			} else if (p->t.type() == Token::category::flash) {
				collect_flash_drive(p->stmt);
			} else if (p->t.type() == Token::category::network) {
				collect_network(p->stmt);
			} else {
				throw std::runtime_error("Unknown controller type");
			}
		} else {
			throw std::runtime_error("Unknown statement");
		}
	}
}

void Program::collect_test(const std::shared_ptr<AST::Test>& test) {
	auto inserted = insert_object(test, tests);
	ordered_tests.push_back(inserted);;
}
void Program::collect_macro(const std::shared_ptr<AST::Macro>& macro) {
	insert_object(macro, macros);
}
void Program::collect_param(const std::shared_ptr<AST::Param>& param_ast) {
	auto param = insert_object(param_ast, params);
	if (stack->vars.count(param->name())) {
		throw std::runtime_error(std::string(param_ast->begin()) + ": Error: param \"" + param->name()
			+ "\" is already defined as a command line argument");
	}
	stack->vars[param->name()] = param->value();
}
void Program::collect_machine(const std::shared_ptr<AST::Controller>& machine) {
	insert_object(machine, machines);
}
void Program::collect_flash_drive(const std::shared_ptr<AST::Controller>& flash) {
	insert_object(flash, flash_drives);
}
void Program::collect_network(const std::shared_ptr<AST::Controller>& network) {
	insert_object(network, networks);
}

void Program::validate_special_params() {
	for (auto& kv: testo_timeout_params) {
		std::string value = stack->resolve_var(kv.first);
		if (!check_if_time_interval(value)) {
			throw std::runtime_error("Can't convert parameter " + kv.first + " value \"" + value + "\" to time interval");
		}
	}
}

void Program::setup_tests_parents() {
	std::string test_spec = config.at("test_spec").get<std::string>();
	std::string exclude = config.at("exclude").get<std::string>();

	for (auto& test: ordered_tests) {
		auto test_name = test->ast_node->name.value();

		if (test_spec.length() && !wildcards::match(test_name, test_spec)) {
			continue;
		}

		if (exclude.length() && wildcards::match(test_name, exclude)) {
			continue;
		}

		setup_test_parents(test);
	}
}

void Program::setup_test_parents(const std::shared_ptr<Test>& test) {
	for (auto& t: all_selected_tests) {
		if (t == test) {
			return;
		}
	}

	all_selected_tests.push_back(test);
	for (auto& parent_token: test->ast_node->parents_tokens) {
		std::string parent_name = parent_token.value();

		if (parent_name == test->name()) {
			throw std::runtime_error(std::string(parent_token.begin()) + ": Error: can't specify test as a parent to itself " + parent_name);
		}

		auto parent = tests.find(parent_name);
		if (parent == tests.end()) {
			throw std::runtime_error(std::string(parent_token.begin()) + ": Error: unknown test: " + parent_name);
		}

		auto result = test->parents.insert(parent->second);
		if (!result.second) {
			throw std::runtime_error(std::string(parent_token.begin()) + ": Error: this test was already specified in parent list " + parent_name);
		}

		setup_test_parents(parent->second);
	}
}

Program* program = nullptr;

}
