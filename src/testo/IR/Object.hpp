
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

namespace IR {

template <typename ASTType>
struct Node {
	Node() = default;
	Node(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
};

template <typename T>
struct Object: Node<T> {
	using ASTType = T;
	std::vector<std::shared_ptr<AST::IMacroCall>> macro_call_stack;
	
	std::string name() const {
		return ast_node->name.value();
	}

	std::shared_ptr<StackNode> new_stack(std::map<std::string, std::string> vars) {
		auto new_node = std::make_shared<StackNode>();
		new_node->parent = stack;
		new_node->vars = std::move(vars);
		return new_node;
	}
};

}
