
#include "TemplateParser.hpp"

namespace template_literals {

std::string Parser::resolve_var(const std::string& var, Register& reg) {
	for (auto it = reg.local_vars.rbegin(); it != reg.local_vars.rend(); ++it) {
		if (it->is_defined(var)) {
			return it->ref(var);
		}
		if (it->is_terminate) {
			break;
		}
	}

	auto found = reg.params.find(var);
	if (found == reg.params.end()) {
		return "";
	}

	return found->second;
}

std::string Parser::resolve(const std::string& input, Register& reg) {
	this->input = input;
	current_pos = Pos(input);

	auto tokens = tokenize();

	std::string result;

	for (auto token: tokens) {
		if (token.type() == Token::category::var_ref) {
			result += resolve_var(token.value().substr(2, token.value().length() - 3), reg);
		} else if (token.type() == Token::category::regular_string) {
			result += token.value();
		} else {
			//should never happen
			throw std::runtime_error(std::string(current_pos) + " -> ERROR: Unknown lexem: " + token.value());
		}
	}

	return result;
}

void Parser::validate_sanity(const std::string& input) {
	this->input = input;
	current_pos = Pos(input);

	auto tokens = tokenize();
}

bool Parser::test_escaped() const {
	if (test_eof(1)) {
		return false;
	}

	return ((input[current_pos] == '$') &&
		(input[current_pos + 1] == '$'));
}

bool Parser::test_var_ref() const {
	if (test_eof(1)) {
		return false;
	}
	return (input[current_pos] == '$' && input[current_pos + 1] == '{');
}

bool Parser::test_id(size_t shift) const {
	return (isalpha(input[current_pos + shift]) ||
		(input[current_pos + shift] == '_'));
}

Token Parser::var_ref() {
	Pos tmp_pos = current_pos;
	std::string value;
	value += input[current_pos];
	current_pos.advance();

	if (test_eof()) {
		throw std::runtime_error(std::string(current_pos) + ": Error: unexpected end of line in var referencing, expected \"{\"");
	}

	if (input[current_pos] != '{') {
		throw std::runtime_error(std::string(current_pos) + ": Error: unexpected symbol in var referencing: " + input[current_pos] + " expected \"{\"");
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
		throw std::runtime_error(std::string(current_pos) + ": Error: unexpected end of line in var referencing, expected \"}\"");
	}

	if (shift == 0) {
		throw std::runtime_error(std::string(tmp_pos) + ": Error: empty var reference");
	}


	if (input[current_pos] != '}') {
		throw std::runtime_error(std::string(current_pos) + ": Error: unexpected symbol in var referencing: " + input[current_pos] + " expected \"}\"");
	}

	value += input[current_pos];
	current_pos.advance();

	return Token(Token::category::var_ref, value, tmp_pos);
}

std::vector<Token> Parser::tokenize() {
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
		if (test_var_ref()) {
			if (string_value.length()) {
				result.push_back(Token(Token::category::regular_string, string_value, string_start));
			}
			result.push_back(var_ref());
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

