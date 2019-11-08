
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

/*struct Expr: public Node {
	Expr():
		Node(select), from(from), where(where), where_expr(where_expr) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};*/

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct SelectStmt: public Node {
	//SelectStmt(const Token& select, const Token& from, const Token& where, std::shared_ptr<Expr> where_expr):
	SelectStmt(const Token& select, const std::vector<Token>& columns, const Token& from_token, const Token& where_token):
		//Node(select), from(from), where(where), where_expr(where_expr) {}
		Node(select), columns(columns), from_token(from_token), where_token(where_token) {}

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
	//std::shared_ptr<Expr> where_expr;
};

}

}
