
#pragma once

#include "Object.hpp"

namespace IR {

template <typename BlockType>
std::shared_ptr<BlockType> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::StmtBlock> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::CmdBlock> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::Action<AST::ActionBlock>> parse_block(const std::vector<Token>& tokens);

struct Macro: Object<AST::Macro> {
	static std::string type_name() { return "macro"; }

	template <typename MacroBodyType>
	std::shared_ptr<AST::MacroBody<MacroBodyType>> get_body() {
		if (!ast_node->body) {
			auto block = parse_block<typename MacroBodyType::BlockType>(ast_node->body_tokens);
			auto body = std::make_shared<MacroBodyType>(block);
			ast_node->body = std::make_shared<AST::MacroBody<MacroBodyType>>(body);
		}
		return std::dynamic_pointer_cast<AST::MacroBody<MacroBodyType>>(ast_node->body);
	}

	void validate();
};

}
