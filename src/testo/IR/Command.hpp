#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

namespace IR {

struct Command {
	Command(std::shared_ptr<AST::Cmd> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<AST::Cmd> ast_node;
	std::shared_ptr<StackNode> stack;


	std::string entity() const;
};



}