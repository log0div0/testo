
#pragma once

#include "Lexer.hpp"
#include "Node.hpp"
#include "Token.hpp"

namespace screen_selection {

struct Parser {
	Parser(const std::string& query): lex(query) {}
	std::shared_ptr<AST::SelectStmt> parse();

private:
	void consume();
	void match(Token::category type);

	Token LT() const;
	Token::category LA() const;

	Token current_token;
	Lexer lex;
};

}
