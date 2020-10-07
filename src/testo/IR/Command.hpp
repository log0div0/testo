#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

namespace IR {

template <typename ASTType>
struct Command {
	Command(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;

	std::string entity() const;
};

struct RegularCommand: Command<AST::RegularCmd> {
	using Command<AST::RegularCmd>::Command;
	std::string entity() const;
};

}