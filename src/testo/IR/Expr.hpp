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
	using Expr<AST::Comparison>::Expr;
	std::string op() const;
	std::string left() const;
	std::string right() const;
	bool calculate() const;
};

struct Check: Expr<AST::Check> {
	using Expr<AST::Check>::Expr;
	SelectExpr select_expr() const;
	TimeInterval timeout() const;
	TimeInterval interval() const;
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
