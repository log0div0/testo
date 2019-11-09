
#pragma once

#include "Pos.hpp"
#include "Token.hpp"
#include <memory>
#include <vector>

namespace screen_selection {

namespace AST {

struct Node {
	Node(const Token& t): t(t) {}
	virtual ~Node() {}
	virtual Pos begin() const = 0;
	virtual Pos end() const = 0;
	virtual operator std::string() const = 0;

	Token t;
};

struct Term: public Node {
	Term(const Token& value):
		Node(value) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};


struct Factor: public Node {
	Factor(std::shared_ptr<Term> left, const Token& op, std::shared_ptr<Term> right):
		Node(Token(Token::category::factor, "factor", left->begin())), left(left), op(op), right(right) {}

	Pos begin() const {
		return left->begin();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return std::string(*left) + op.value() + std::string(*right);
	}

	std::shared_ptr<Term> left;
	Token op;
	std::shared_ptr<Term> right;
};

struct IExpr: public Node {
	using Node::Node;
};

//Factor, Unary, Binary
template <typename ExprType>
struct Expr: IExpr {
	Expr(std::shared_ptr<ExprType> expr):
		IExpr(expr->t),
		expr(expr) {}

	Pos begin() const {
		return expr->begin();
	}

	Pos end() const {
		return expr->end();
	}

	operator std::string() const {
		return std::string(*expr);
	}

	std::shared_ptr<ExprType> expr;
};

struct UnOp: public Node {
	UnOp(const Token& op, std::shared_ptr<IExpr> expr):
		Node(op), expr(expr) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return expr->end();
	}

	operator std::string() const {
		return t.value() + std::string(*expr);
	}

	std::shared_ptr<IExpr> expr;
};

struct BinOp: public Node {
	BinOp(std::shared_ptr<IExpr> left, const Token& op, std::shared_ptr<IExpr> right):
		Node(op), left(left), right(right) {}

	Pos begin() const {
		return left->begin();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return std::string(*left) + t.value() + std::string(*right);
	}

	std::shared_ptr<IExpr> left;
	std::shared_ptr<IExpr> right;	
};

struct ParentedExpr: public Node {
	ParentedExpr(const Token& lparen, std::shared_ptr<IExpr> expr, const Token& rparen):
		Node(lparen), expr(expr), rparen(rparen) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return rparen.pos();
	}

	operator std::string() const {
		return t.value() + std::string(*expr) + rparen.value();
	}

	std::shared_ptr<IExpr> expr;
	Token rparen;
};

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct SelectStmt: public Node {
	//SelectStmt(const Token& select, const Token& from, const Token& where, std::shared_ptr<Expr> where_expr):
	SelectStmt(
			const Token& select,
			const std::vector<Token>& columns,
			const Token& from_token,
			const Token& where_token,
			std::shared_ptr<IExpr> where_expr):
		Node(select), columns(columns), from_token(from_token), where_token(where_token), where_expr(where_expr) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}

	std::vector<Token> columns;
	Token from_token;
	Token from_table;
	Token where_token;
	std::shared_ptr<IExpr> where_expr;
};

}

}
