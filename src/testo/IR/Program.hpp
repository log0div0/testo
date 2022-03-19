
#pragma once

#include "../Configs.hpp"
#include "Test.hpp"
#include "Macro.hpp"
#include "Param.hpp"
#include <unordered_set>
#include <unordered_map>

namespace IR {

struct Program {
	Program(const std::shared_ptr<AST::Program>& ast, const ProgramConfig& config);
	~Program();

	Program(const Program& other) = delete;
	Program& operator=(const Program& other) = delete;
	Program(Program&& other) = delete;
	Program& operator=(Program&& other) = delete;

	void validate();
	void run();

	const ProgramConfig& config;

private:
	MacroCallStack current_macro_call_stack;

	std::unordered_map<std::string, std::shared_ptr<Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<Macro>> macros;
	std::unordered_map<std::string, std::shared_ptr<Param>> params;
	std::unordered_map<std::string, std::shared_ptr<Machine>> machines;
	std::unordered_map<std::string, std::shared_ptr<FlashDrive>> flash_drives;
	std::unordered_map<std::string, std::shared_ptr<Network>> networks;
	std::unordered_map<std::string, std::shared_ptr<Controller>> controllers;

public:
	std::shared_ptr<Macro> get_macro_or_throw(const std::string& name);
	std::shared_ptr<Param> get_param_or_throw(const std::string& name);
	std::shared_ptr<Machine> get_machine_or_throw(const std::string& name);
	std::shared_ptr<FlashDrive> get_flash_drive_or_throw(const std::string& name);
	std::shared_ptr<Network> get_network_or_throw(const std::string& name);

	std::shared_ptr<Test> get_test_or_null(const std::string& name);
	std::shared_ptr<Macro> get_macro_or_null(const std::string& name);
	std::shared_ptr<Param> get_param_or_null(const std::string& name);
	std::shared_ptr<Machine> get_machine_or_null(const std::string& name);
	std::shared_ptr<FlashDrive> get_flash_drive_or_null(const std::string& name);
	std::shared_ptr<Network> get_network_or_null(const std::string& name);

	std::vector<std::shared_ptr<Test>> ordered_tests;
	std::vector<std::shared_ptr<Test>> all_selected_tests;
	std::shared_ptr<StackNode> stack;
	std::unordered_set<std::shared_ptr<IR::Macro>> visited_macros;

	std::string resolve_top_level_param(const std::string& name) const;
	template <typename T>
	std::shared_ptr<T> get_top_level_param_ast(const std::string& name) const {
		static std::map<std::string, std::shared_ptr<T>> cache;
		auto it = cache.find(name);
		if (it != cache.end()) {
			return it->second;
		}
		std::string value = resolve_top_level_param(name);
		std::shared_ptr<T> p = T::from_string(value);
		cache[name] = p;
		return p;
	}

private:
	friend struct IR::MacroCall;

	void setup_stack();

	void collect_top_level_objects(const std::shared_ptr<AST::Program>& ast);
	void visit_statement_block(const std::shared_ptr<AST::Block<AST::Stmt>>& stmt_block);
	void visit_stmt(const std::shared_ptr<AST::Stmt>& stmt);
	void visit_macro(std::shared_ptr<IR::Macro> macro);
	void visit_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::Block<AST::Stmt>>& macro_body);
	void collect_test(const std::shared_ptr<AST::Test>& ast);
	void collect_macro(const std::shared_ptr<AST::Macro>& ast);
	void collect_param(const std::shared_ptr<AST::Param>& ast);
	void collect_machine(const std::shared_ptr<AST::Controller>& ast);
	void collect_flash_drive(const std::shared_ptr<AST::Controller>& ast);
	void collect_network(const std::shared_ptr<AST::Controller>& ast);

	bool validate_test_name(const std::string& name, const std::vector<std::pair<bool, std::string>>& patterns) const;
	void validate_special_params();

	void setup_tests_parents();
	void setup_test_parents(const std::shared_ptr<Test>& test);

private:
	template <typename T>
	std::shared_ptr<T> insert_object(const std::shared_ptr<typename T::ASTType>& ast_node, std::unordered_map<std::string, std::shared_ptr<T>>& map) {
		auto t = std::make_shared<T>();
		t->ast_node = ast_node;
		t->stack = stack;
		t->macro_call_stack = current_macro_call_stack;
		auto inserted = map.insert({t->name(), t});
		if (!inserted.second) {
			std::stringstream ss;
			ss << std::string(ast_node->begin()) + ": Error: " + T::type_name() + " \"" + t->name() + "\" is already defined" << std::endl << std::endl;
			ss << inserted.first->second->macro_call_stack;
			ss << std::string(inserted.first->second->ast_node->begin()) << ": note: previous declaration was here";

			throw Exception(ss.str());
		}
		return t;
	}

	template <typename T>
	std::shared_ptr<T> insert_controller(const std::shared_ptr<typename T::ASTType>& ast_node, std::unordered_map<std::string, std::shared_ptr<T>>& map) {
		auto controller = insert_object(ast_node, map);

		auto inserted = controllers.insert({controller->name(), controller});
		if (!inserted.second) {
			throw ExceptionWithPos(ast_node->begin(), "Error: " + inserted.first->second->type() + " \"" + controller->name() + "\" is already defined here: " +
				std::string(inserted.first->second->ast_node->begin()));
		}

		return controller;
	}

	template <typename T>
	std::shared_ptr<T> get_or_null(const std::string& name, const std::unordered_map<std::string, std::shared_ptr<T>>& map) {
		auto it = map.find(name);
		if (it == map.end()) {
			return nullptr;
		}
		return it->second;
	}

	template <typename T>
	std::shared_ptr<T> get_or_throw(const std::string& name, const std::unordered_map<std::string, std::shared_ptr<T>>& map) {
		auto it = map.find(name);
		if (it == map.end()) {
			throw std::runtime_error(T::type_name() + " with name " + name + " not found");
		}
		return it->second;
	}
};

extern Program* program;

}
