
#include "Parser.hpp"

namespace screen_selection {

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

std::shared_ptr<AST::ExecPlan> Parser::parse() {
	return nullptr;
}

}
