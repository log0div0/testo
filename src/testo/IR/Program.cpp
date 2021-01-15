
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

void Program::visit_stmt(const std::shared_ptr<AST::IStmt>& stmt) {
	if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
		collect_test(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::MacroCall>>(stmt)) {
		visit_macro_call(p->stmt);
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

void Program::visit_statement_block(const std::shared_ptr<AST::StmtBlock>& stmt_block) {
	for (auto stmt: stmt_block->stmts) {
		visit_stmt(stmt);
	}
}

void Program::collect_test(const std::shared_ptr<AST::Test>& test) {
	auto inserted = insert_object(test, tests);
	ordered_tests.push_back(inserted);;
	inserted->macro_call_stack = current_macro_call_stack;
}

void Program::visit_macro(std::shared_ptr<IR::Macro> macro) {
	auto result = visited_macros.insert(macro);
	if (!result.second) {
		return;
	}

	StackPusher<Program> new_ctx(this, macro->stack);

	for (size_t i = 0; i < macro->ast_node->args.size(); ++i) {
		for (size_t j = i + 1; j < macro->ast_node->args.size(); ++j) {
			if (macro->ast_node->args[i]->name() == macro->ast_node->args[j]->name()) {
				throw std::runtime_error(std::string(macro->ast_node->args[j]->begin()) + ": Error: duplicate macro arg: " + macro->ast_node->args[j]->name());
			}
		}
	}

	bool has_default = false;
	for (auto arg: macro->ast_node->args) {
		if (arg->default_value) {
			has_default = true;
			continue;
		}

		if (has_default && !arg->default_value) {
			throw std::runtime_error(std::string(arg->begin()) + ": Error: default value must be specified for macro arg " + arg->name());
		}
	}
}

void Program::visit_macro_call(const std::shared_ptr<AST::MacroCall>& macro_call) {
	auto macro = get_macro_or_null(macro_call->name().value());
	if (!macro) {
		throw std::runtime_error(std::string(macro_call->begin()) + ": Error: unknown macro: " + macro_call->name().value());
	}

	visit_macro(macro);

	uint32_t args_with_default = 0;

	for (auto arg: macro->ast_node->args) {
		if (arg->default_value) {
			args_with_default++;
		}
	}

	if (macro_call->args.size() < macro->ast_node->args.size() - args_with_default) {
		throw std::runtime_error(fmt::format("{}: Error: expected at least {} args, {} provided", std::string(macro_call->begin()),
			macro->ast_node->args.size() - args_with_default, macro_call->args.size()));
	}

	if (macro_call->args.size() > macro->ast_node->args.size()) {
		throw std::runtime_error(fmt::format("{}: Error: expected at most {} args, {} provided", std::string(macro_call->begin()),
			macro->ast_node->args.size(), macro_call->args.size()));
	}

	std::map<std::string, std::string> vars;

	for (size_t i = 0; i < macro_call->args.size(); ++i) {
		try {
			auto value = template_parser.resolve(macro_call->args[i]->text(), stack);
			vars[macro->ast_node->args[i]->name()] = value;
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro_call->args[i]->begin(), macro_call->args[i]->text()));
		}
	}

	for (size_t i = macro_call->args.size(); i < macro->ast_node->args.size(); ++i) {
		try {
			auto value = template_parser.resolve(macro->ast_node->args[i]->default_value->text(), stack);
			vars[macro->ast_node->args[i]->name()] = value;
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro->ast_node->args[i]->default_value->begin(), macro->ast_node->args[i]->default_value->text()));
		}
	}

	StackPusher<Program> new_ctx(this, macro->new_stack(vars));
	current_macro_call_stack.push_back(macro_call);

	if (!macro->ast_node->body) {
		Parser parser(macro->ast_node->body_tokens);

		auto stmt_block = parser.stmt_block();
		auto body = std::shared_ptr<AST::MacroBodyStmt>(new AST::MacroBodyStmt(stmt_block));
		macro->ast_node->body = std::shared_ptr<AST::MacroBody<AST::MacroBodyStmt>>(new AST::MacroBody<AST::MacroBodyStmt>(body));
	}

	try {
		auto p = std::dynamic_pointer_cast<AST::MacroBody<AST::MacroBodyStmt>>(macro->ast_node->body);
		if (p == nullptr) {
			throw std::runtime_error(std::string(macro_call->begin()) + ": Error: the \"" + macro_call->name().value() + "\" macro does not contain statements, as expected");
		}

		visit_statement_block(p->macro_body->stmt_block);
		current_macro_call_stack.pop_back();
	} catch (const std::exception& error) {
		std::throw_with_nested(MacroException(macro_call));
	}
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
