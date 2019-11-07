
#pragma once

#include "Pos.hpp"
#include "Token.hpp"
#include <memory>

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

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct ExecPlan: public Node {
	ExecPlan():
		Node(Token(Token::category::exec_plan, "exec_plan", Pos())) {}

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

}

}
