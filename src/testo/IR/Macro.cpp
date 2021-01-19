
#include "Macro.hpp"
#include "../Parser.hpp"

namespace IR {

template <>
std::shared_ptr<AST::StmtBlock> parse_block(const std::vector<Token>& tokens) {
	return Parser(tokens).stmt_block();
}

template <>
std::shared_ptr<AST::CmdBlock> parse_block(const std::vector<Token>& tokens) {
	return Parser(tokens).command_block();
}

template <>
std::shared_ptr<AST::Action<AST::ActionBlock>> parse_block(const std::vector<Token>& tokens) {
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

}
