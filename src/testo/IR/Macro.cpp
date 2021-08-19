
#include "Macro.hpp"
#include "Program.hpp"
#include "../Parser.hpp"
#include "../Exceptions.hpp"

namespace IR {

template <>
std::shared_ptr<AST::Block<AST::Stmt>> parse_block(const std::vector<Token>& tokens) {
	return Parser(tokens).stmt_block();
}

template <>
std::shared_ptr<AST::Block<AST::Cmd>> parse_block(const std::vector<Token>& tokens) {
	return Parser(tokens).command_block();
}

template <>
std::shared_ptr<AST::Block<AST::Action>> parse_block(const std::vector<Token>& tokens) {
	return Parser(tokens).action_block();
}

void Macro::validate() {
	for (size_t i = 0; i < ast_node->args.size(); ++i) {
		for (size_t j = i + 1; j < ast_node->args.size(); ++j) {
			if (ast_node->args[i]->name() == ast_node->args[j]->name()) {
				throw std::runtime_error(std::string(ast_node->args[j]->begin()) + ": Error: duplicate macro arg: " + ast_node->args[j]->name());
			}
		}
	}

	bool has_default = false;
	for (auto arg: ast_node->args) {
		if (arg->default_value) {
			has_default = true;
			continue;
		}

		if (has_default && !arg->default_value) {
			throw std::runtime_error(std::string(arg->begin()) + ": Error: default value must be specified for macro arg " + arg->name());
		}
	}
}

const std::shared_ptr<IR::Macro> MacroCall::get_macro() const {
	auto macro = program->get_macro_or_null(ast_node->name.value());
	if (!macro) {
		throw std::runtime_error(std::string(ast_node->name.begin()) + ": Error: unknown macro: " + ast_node->name.value());
	}
	return macro;
}

std::vector<std::pair<std::string, std::string>> MacroCall::args() const {
	std::vector<std::pair<std::string, std::string>> args;
	const std::shared_ptr<IR::Macro> macro = get_macro();

	for (size_t i = 0; i < ast_node->args.size(); ++i) {
		try {
			auto value = template_literals::Parser().resolve(ast_node->args[i]->text(), stack);
			args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(ast_node->args[i]->begin(), ast_node->args[i]->text()));
		}
	}

	for (size_t i = ast_node->args.size(); i < macro->ast_node->args.size(); ++i) {
		try {
			auto value = template_literals::Parser().resolve(macro->ast_node->args[i]->default_value->text(), stack);
			args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro->ast_node->args[i]->default_value->begin(), macro->ast_node->args[i]->default_value->text()));
		}
	}

	return args;
}

std::map<std::string, std::string> MacroCall::vars() const {
	std::map<std::string, std::string> vars;

	for (auto& kv: args()) {
		vars[kv.first] = kv.second;
	}

	return vars;
}

}
