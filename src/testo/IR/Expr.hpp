#pragma once

#include "Base.hpp"

namespace IR {

template <typename ASTType>
struct Expr: Node<ASTType> {
	using Node<ASTType>::Node;
};

struct Defined: Expr<AST::Defined> {
	using Expr<AST::Defined>::Expr;
	bool is_defined() const;
	std::string var() const;
};

struct Comparison: Expr<AST::Comparison> {
	Comparison(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Expr(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}
	std::string op() const;
	String left() const;
	String right() const;
	bool can_resolve_variables() const;
	bool calculate() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct Check: Expr<AST::Check> {
	Check(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Expr(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	SelectExpr select_expr() const;
	TimeInterval timeout() const;
	TimeInterval interval() const;

private:
	std::shared_ptr<VarMap> var_map;
};

template <typename ASTType>
struct CounterList: Node<ASTType> {
	using Node<ASTType>::Node;
};

struct Range: CounterList<AST::Range> {
	using CounterList<AST::Range>::CounterList;
	std::vector<std::string> values() const;
	int32_t r1() const;
	int32_t r2() const;
};

}
