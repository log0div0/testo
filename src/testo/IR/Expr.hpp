#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

namespace IR {

template <typename ASTType>
struct Factor {
	Factor(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
};

struct Defined: Factor<AST::Defined> {
	using Factor<AST::Defined>::Factor;
	bool is_defined() const;
	std::string var() const;
};

struct Comparison: Factor<AST::Comparison> {
	using Factor<AST::Comparison>::Factor;
	std::string op() const;
	std::string left() const;
	std::string right() const;
};

template <typename ASTType>
struct CounterList {
	CounterList(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
};

struct Range: CounterList<AST::Range> {
	using CounterList<AST::Range>::CounterList;
	std::vector<std::string> values() const;
	std::string r1() const;
	std::string r2() const;
};

}
