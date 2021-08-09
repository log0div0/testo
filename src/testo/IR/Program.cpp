
#include "Program.hpp"
#include "../backends/Environment.hpp"
#include "nn/OnnxRuntime.hpp"
#include <fmt/format.h>
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"
#include "../Parser.hpp"
#include <wildcards.hpp>

namespace IR {

bool TestNameFilter::validate_test_name(const std::string& name) const {
	switch (type) {
		case Type::test_spec:
			return wildcards::match(name, pattern);
		case Type::exclude:
			return !wildcards::match(name, pattern);
		default:
			throw std::runtime_error("Should not be there");
	}
}

void to_json(nlohmann::json& j, const TestNameFilter& filter) {
	switch (filter.type) {
		case TestNameFilter::Type::test_spec:
			j["type"] = "test_spec";
			break;
		case TestNameFilter::Type::exclude:
			j["type"] = "exclude";
			break;
		default:
			throw std::runtime_error("Should not be there");
	}
	j["pattern"] = filter.pattern;
}

bool ProgramConfig::validate_test_name(const std::string& name) const {
	for (auto& filter: test_name_filters) {
		if (!filter.validate_test_name(name)) {
			return false;
		}
	}
	return true;
}

void ProgramConfig::validate() const {
	if (!fs::exists(target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + target);
	}

	std::set<std::string> unique_param_names;

	for (size_t i = 0; i < params_names.size(); ++i) {
		auto result = unique_param_names.insert(params_names[i]);
		if (!result.second) {
			throw std::runtime_error("Error: param \"" + params_names[i] + "\" is defined multiple times as a command line argument");
		}
	}

	VisitorSemanticConfig::validate();
	VisitorInterpreterConfig::validate();
}

void ProgramConfig::dump(nlohmann::json& j) const {
	VisitorSemanticConfig::dump(j);
	VisitorInterpreterConfig::dump(j);

	j["target"] = target;
	j["test_name_filters"] = test_name_filters;
	auto params = nlohmann::json::object();
	for (size_t i = 0; i < params_names.size(); ++i) {
		params[params_names.at(i)] = params_values.at(i);
	}
	j["params"] = params;
	j["use_cpu"] = use_cpu;
}

Program::Program(const std::shared_ptr<AST::Program>& ast, const ProgramConfig& config_): config(config_) {
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
#ifdef USE_CUDA
	nn::onnx::Runtime onnx_runtime(config.use_cpu);
#else
	nn::onnx::Runtime onnx_runtime;
#endif
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
	return get_or_null(name, machines);
}

std::shared_ptr<FlashDrive> Program::get_flash_drive_or_null(const std::string& name) {
	return get_or_null(name, flash_drives);
}

std::shared_ptr<Network> Program::get_network_or_null(const std::string& name) {
	return get_or_null(name, networks);
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
	for (size_t i = 0; i < config.params_names.size(); ++i) {
		stack->vars[config.params_names.at(i)] = config.params_values.at(i);
	}
}

void Program::collect_top_level_objects(const std::shared_ptr<AST::Program>& ast) {
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
		if (p->t.type() == Token::category::machine) {
			collect_machine(p);
		} else if (p->t.type() == Token::category::flash) {
			collect_flash_drive(p);
		} else if (p->t.type() == Token::category::network) {
			collect_network(p);
		} else {
			throw std::runtime_error("Unknown controller type");
		}
	} else {
		throw std::runtime_error("Unknown statement");
	}
}

void Program::visit_statement_block(const std::shared_ptr<AST::StmtBlock>& stmt_block) {
	for (auto stmt: stmt_block->stmts) {
		if (auto p = std::dynamic_pointer_cast<AST::Macro>(stmt)) {
			throw Exception(std::string(stmt->begin()) + ": Error: nested macro declarations are not supported");
		} else if (auto p = std::dynamic_pointer_cast<AST::Param>(stmt)) {
			throw Exception(std::string(stmt->begin()) + ": Error: param declaration inside macros is not supported");
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
	macro_call.visit_semantic<AST::MacroBodyStmt>(this);
	current_macro_call_stack.pop_back();
}

void Program::visit_macro_body(const std::shared_ptr<AST::MacroBodyStmt>& macro_body) {
	visit_statement_block(macro_body->stmt_block);
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
	insert_controller(machine, machines);
}
void Program::collect_flash_drive(const std::shared_ptr<AST::Controller>& flash) {
	insert_controller(flash, flash_drives);
}
void Program::collect_network(const std::shared_ptr<AST::Controller>& network) {
	insert_controller(network, networks);
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
			throw std::runtime_error(std::string(test->ast_node->parents[i]->begin()) + ": Error: can't specify test as a parent to itself " + parent_name);
		}

		auto parent = tests.find(parent_name);
		if (parent == tests.end()) {
			throw std::runtime_error(std::string(test->ast_node->parents[i]->begin()) + ": Error: unknown test: " + parent_name);
		}

		auto result = test->parents.insert(parent->second);
		if (!result.second) {
			throw std::runtime_error(std::string(test->ast_node->parents[i]->begin()) + ": Error: this test was already specified in parent list " + parent_name);
		}

		setup_test_parents(parent->second);
	}
}

Program* program = nullptr;

}
