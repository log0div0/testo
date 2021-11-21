
#include "TemplateLiterals.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include <stdexcept>

namespace template_literals {

Resolver::Resolver(const std::string& input_): input(input_) {
	current_pos = Pos(input);
	tokens = tokenize();
}

std::string Resolver::resolve(const std::shared_ptr<const StackNode>& stack) const {
	std::string result;

	for (auto token: tokens) {
		if (token.type() == Token::category::param_ref) {
			result += stack->find_and_resolve_var(token.value().substr(2, token.value().length() - 3));
		} else if (token.type() == Token::category::var_ref) {
			// TODO
		} else if (token.type() == Token::category::regular_string) {
			result += token.value();
		} else {
			//should never happen
			throw ExceptionWithPos(current_pos, "Error: Unknown lexem: " + token.value());
		}
	}

	return result;
}

bool Resolver::has_variables() const {
	for (auto token: tokens) {
		if (token.type() == Token::category::var_ref) {
			return true;
		}
	}
	return false;
}

bool Resolver::can_resolve_variables() const {
	if (!has_variables()) {
		return true;
	}
	return IR::program && IR::program->var_set;
}

bool Resolver::test_escaped() const {
	if (test_eof(1)) {
		return false;
	}

	return ((input[current_pos] == '$') &&
		(input[current_pos + 1] == '$'));
}

bool Resolver::test_ref() const {
	if (test_eof(1)) {
		return false;
	}
	char cur = input[current_pos];
	char next = input[current_pos+1];
	return (cur == '$' && (next == '{' || next == '<'));
}

bool Resolver::test_id(size_t shift) const {
	return (isalpha(input[current_pos + shift]) ||
		(input[current_pos + shift] == '_'));
}

Token Resolver::ref() {
	Pos tmp_pos = current_pos;
	std::string value;
	value += input[current_pos];
	current_pos.advance();

	Token::category category;
	char close_char;
	if (input[current_pos] == '{') {
		category = Token::category::param_ref;
		close_char = '}';
	} else if (input[current_pos] == '<') {
		category = Token::category::var_ref;
		close_char = '>';
	} else {
		throw std::runtime_error("Should not be there");
	}

	value += input[current_pos];
	current_pos.advance();

	size_t shift = 0;

	while (!test_eof(shift) && (test_id(shift) || isdigit(input[current_pos + shift]))) {
		value += input[current_pos + shift];
		shift++;
	}

	current_pos.advance(shift);

	if (test_eof()) {
		throw ExceptionWithPos(current_pos, std::string("Error: unexpected end of line in var referencing, expected \"") + close_char + "\"");
	}

	if (shift == 0) {
		throw ExceptionWithPos(tmp_pos, "Error: empty var reference");
	}

	if (input[current_pos] != close_char) {
		throw ExceptionWithPos(current_pos, std::string("Error: unexpected symbol in var referencing: ") + input[current_pos] + " expected \"" + close_char + "\"");
	}

	value += input[current_pos];
	current_pos.advance();

	return Token(category, value, tmp_pos);
}

std::vector<Token> Resolver::tokenize() {
	std::vector<Token> result;

	std::string string_value;
	Pos string_start = current_pos;
	//Well, this could seem a little crooked,
	//But actually it's just a lesser code for only two tokens:
	//Regular string and var reference
	//Basically we just extend strings a little by little
	//And interrupt when a reference occurs
	//This would work just fine while we have only two tokens
	while (!test_eof()) {
		if (test_ref()) {
			if (string_value.length()) {
				result.push_back(Token(Token::category::regular_string, string_value, string_start));
			}
			result.push_back(ref());
			string_value = "";
			string_start = current_pos;
			continue;
		}

		if (test_escaped()) {
			current_pos.advance();
		}

		string_value += input[current_pos];
		current_pos.advance();
	}

	if (string_value.length()) {
		result.push_back(Token(Token::category::regular_string, string_value, string_start));
	}

	return result;
}


}

