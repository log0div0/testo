
#include "Parser.hpp"

namespace screen_selection {

Parser::Parser(const std::string& query): lex(query) {
	consume();
}

void Parser::consume() {
	current_token = lex.get_next_token();
}

void Parser::match(Token::category type) {
	if (LA() == type) {
		consume();
	} else {
		throw std::runtime_error(std::string(LT().pos()) +
			": unexpected token \"" +
			LT().value() + "\", expected: " + Token::type_to_string(type)); //TODO: more informative what we expected
	}
}

void Parser::match(const std::vector<Token::category> types) {
	for (auto type: types) {
		if (LA() == type) {
			consume();
			return;
		}
	}

	throw std::runtime_error(std::string(LT().pos()) +
			": unexpected token \"" +
			LT().value() + "\""); //TODO: more informative what we expected
}

Token Parser::LT() const {
	return current_token;
}

Token::category Parser::LA() const {
	return LT().type();
}

std::shared_ptr<AST::Term> Parser::term() {
	auto value = LT();

	match({Token::category::id, Token::category::dbl_quoted_string, Token::category::number});

	return std::shared_ptr<AST::Term>(new AST::Term(value));
}

std::shared_ptr<AST::Factor> Parser::factor() {
	auto left = term();

	auto op = LT();

	//for now we support only equals and not equals
	match({Token::category::equals, Token::category::not_equals});

	auto right = term();

	return std::shared_ptr<AST::Factor>(new AST::Factor(left, op, right));
}

std::shared_ptr<AST::Expr<AST::UnOp>> Parser::unop() {
	auto op = LT();

	match(Token::category::not_);

	auto expression = expr();

	auto unop = std::shared_ptr<AST::UnOp>(new AST::UnOp(op, expression));
	return std::shared_ptr<AST::Expr<AST::UnOp>>(new AST::Expr<AST::UnOp>(unop));
}

std::shared_ptr<AST::Expr<AST::BinOp>> Parser::binop(std::shared_ptr<AST::IExpr> left) {
	auto op = LT();

	match({Token::category::or_, Token::category::and_});

	auto right = expr();

	auto binop = std::shared_ptr<AST::BinOp>(new AST::BinOp(left, op, right));
	return std::shared_ptr<AST::Expr<AST::BinOp>>(new AST::Expr<AST::BinOp>(binop));
}

std::shared_ptr<AST::Expr<AST::ParentedExpr>> Parser::parented_expr() {
	auto lparen = LT();
	match(Token::category::lparen);

	auto expression = expr();

	auto rparen = LT();
	match(Token::category::rparen);
	auto parented_expr = std::shared_ptr<AST::ParentedExpr>(new AST::ParentedExpr(lparen, expression, rparen));
	return std::shared_ptr<AST::Expr<AST::ParentedExpr>>(new AST::Expr<AST::ParentedExpr>(parented_expr));
}

std::shared_ptr<AST::IExpr> Parser::expr() {
	if (LA() == Token::category::not_) {
		return unop();
	}

	if (LA() == Token::category::lparen) {
		return parented_expr();
	}

	auto left = std::shared_ptr<AST::Expr<AST::Factor>>(new AST::Expr<AST::Factor>(factor()));

	if ((LA() == Token::category::and_) ||
		(LA() == Token::category::or_)) {
		return binop(left);
	} else {
		return left;
	}
}

std::shared_ptr<AST::SelectStmt> Parser::parse() {
	auto select = LT();

	match(Token::category::select);

	std::vector<Token> columns;

	if (LA() == Token::category::asterisk) {
		columns.push_back(LT());
		match(Token::category::asterisk);
	} else {
		columns.push_back(LT());
		match(Token::category::id);

		while (LA() == Token::category::comma) {
			match(Token::category::comma);
			columns.push_back(LT());
			match(Token::category::id);
		}
	}

	auto from = LT();
	match(Token::category::from);
	auto from_table = LT();
	match(Token::category::id);

	auto where_token = LT();
	match(Token::category::where);

	auto where_expr = expr();

	return std::shared_ptr<AST::SelectStmt>(new AST::SelectStmt(select, columns, from, from_table, where_token, where_expr));
}

}
