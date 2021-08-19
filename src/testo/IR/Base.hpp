
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

struct StringTokenUnion: Node<AST::StringTokenUnion> {
	using Node<AST::StringTokenUnion>::Node;
	std::string resolve() const;
};

struct OptionSeq : Node<AST::OptionSeq> {
	using Node<AST::OptionSeq>::Node;
	std::string resolve(const std::string& name, const std::string& fallback_param) const;
};

}
