
#pragma once
#include "Pos.hpp"
#include "Token.hpp"
#include <string>
#include <memory>
#include <algorithm>

namespace screen_selection {

struct Lexer {
	Lexer(std::string input_) {
		std::transform(input_.begin(), input_.end(), input_.begin(), toupper);
		this->input = std::shared_ptr<std::string>(new std::string(input_));
		current_pos = Pos(this->input);
	}

	Token get_next_token();

private:
	bool test_eof(size_t shift = 0) const { return ((current_pos + shift) >= input->length()); }
	bool test_newline() const { return ((*input)[current_pos] == '\n'); }
	bool test_digit(size_t shift = 0) const { return isdigit((*input)[current_pos + shift]); }

	bool test_plus() const { return ((*input)[current_pos] == '+'); }
	bool test_minus() const { return ((*input)[current_pos] == '-'); }

	bool test_number() const {
		if (test_digit()) {
			return true;
		}

		if (test_eof(1)) {
			return false;
		}

		if (test_plus() || test_minus()) {
			return test_digit(1);
		}
		return false;
	}

	bool test_id(size_t shift = 0) const {
		return (isalpha((*input)[current_pos + shift]) ||
			((*input)[current_pos + shift] == '_'));
	}

	bool test_space() const {
		char c = (*input)[current_pos];

		return test_newline() || ((c == ' ') ||
			(c == '\r') ||
			(c == '\t'));
	}

	bool test_dbl_quote() const { return ((*input)[current_pos] == '\"'); }
	bool test_assign() const { return (*input)[current_pos] == '='; }
	bool test_comma() const { return ((*input)[current_pos] == ','); }
	bool test_asterisk() const { return ((*input)[current_pos] == '*'); }
	bool test_lparen() const { return ((*input)[current_pos] == '('); }
	bool test_rparen() const { return ((*input)[current_pos] == ')'); }

	bool test_escaped_character() const { return  ((*input)[current_pos] == '\\'); }
	char escaped_character();

	void skip_spaces();

	Token number();
	Token id();
	Token select();
	Token from();
	Token where();
	Token dbl_quoted_string();
	Token assign();
	Token comma();
	Token asterisk();
	Token lparen();
	Token rparen();

	Token not_();
	Token and_();
	Token or_();

	Pos current_pos;
	std::shared_ptr<std::string> input;
};

}


