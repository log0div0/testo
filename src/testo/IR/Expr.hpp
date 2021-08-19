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
	std::string timeout() const;
	std::string interval() const;
};

template <typename ASTType>
struct CounterList: Node<ASTType> {
	using Node<ASTType>::Node;
};

struct Range: CounterList<AST::Range> {
	using CounterList<AST::Range>::CounterList;
	std::vector<std::string> values() const;
	std::string r1() const;
	std::string r2() const;
};

}
