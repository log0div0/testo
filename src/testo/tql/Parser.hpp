
#pragma once

#include "Lexer.hpp"
#include "Node.hpp"
#include "Token.hpp"

namespace tql {

struct Parser {
	Parser(const std::string& query);
	std::shared_ptr<AST::SelectStmt> parse();

private:
	void consume();
	void match(Token::category type);
	void match(std::vector<Token::category> types);

	Token LT() const;
	Token::category LA() const;

	std::shared_ptr<AST::Term> term();
	std::shared_ptr<AST::Factor> factor();
	std::shared_ptr<AST::IExpr> expr();
	std::shared_ptr<AST::Expr<AST::UnOp>> unop();
	std::shared_ptr<AST::Expr<AST::BinOp>> binop(std::shared_ptr<AST::IExpr> left);
	std::shared_ptr<AST::Expr<AST::ParentedExpr>> parented_expr();

	Token current_token;
	Lexer lex;
};

}
