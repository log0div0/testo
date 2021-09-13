
#pragma once

#include "Base.hpp"

namespace IR {

template <typename T>
struct Object: Node<T> {
	using ASTType = T;
	std::vector<std::shared_ptr<AST::IMacroCall>> macro_call_stack;
	
	std::string name() const {
		return this->ast_node->name.value();
	}

	std::shared_ptr<StackNode> new_stack(std::map<std::string, std::string> vars) {
		auto new_node = std::make_shared<StackNode>();
		new_node->parent = this->stack;
		new_node->vars = std::move(vars);
		return new_node;
	}
};

}
