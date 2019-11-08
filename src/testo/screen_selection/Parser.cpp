
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

Token Parser::LT() const {
	return current_token;
}

Token::category Parser::LA() const {
	return LT().type();
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

	return std::shared_ptr<AST::SelectStmt>(new AST::SelectStmt(select, columns, from, where_token));
}

}
