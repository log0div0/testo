
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

namespace IR {

template <typename T>
struct Object {
	using ASTType = T;
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
	std::vector<std::shared_ptr<AST::MacroCall>> macro_call_stack;
	
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
