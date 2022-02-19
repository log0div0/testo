
#pragma once

#include "Base.hpp"
#include <ostream>

namespace IR {

struct MacroCallStack: std::vector<std::shared_ptr<AST::IMacroCall>> {

};

inline std::ostream& operator<<(std::ostream& stream, const MacroCallStack& macro_call_stack) {
	for (auto macro_call: macro_call_stack) {
		stream << std::string(macro_call->begin()) + std::string(": In a macro call ") << macro_call->to_string() << std::endl;
	}
	return stream;
}

template <typename T>
struct Object: Node<T> {
	using ASTType = T;
	MacroCallStack macro_call_stack;
	
	std::string name() const {
		return this->ast_node->name.value();
	}

	std::shared_ptr<StackNode> new_stack(std::map<std::string, std::string> params) {
		auto new_node = std::make_shared<StackNode>();
		new_node->parent = this->stack;
		new_node->params = std::move(params);
		return new_node;
	}
};

}
