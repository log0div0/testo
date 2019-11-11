
#include "Lexer.hpp"
#include <algorithm>

namespace screen_selection {

void Lexer::skip_spaces() {
	while (test_space() && !test_eof()) {
		current_pos.advance();
	}
}

Token Lexer::number() {
	Pos tmp_pos = current_pos;
	std::string value;

	bool is_signed = false;

	if (test_plus() || test_minus()) {
		is_signed = true;
		value += (*input)[current_pos];
		current_pos.advance();
	}

	while (test_digit() && !test_eof()) {
		value += (*input)[current_pos];
		current_pos.advance();
	}

	if (test_eof()) {
		return Token(Token::category::number, value, tmp_pos);
	}

	if (test_id()) {
		throw std::runtime_error(std::string(tmp_pos) + " -> ERROR: ID can't start with a number");
	} else {
		return Token(Token::category::number, value, tmp_pos);
	}
}


Token Lexer::comma() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::comma, ",", tmp_pos);
}

Token Lexer::equals() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::equals, "=", tmp_pos);
}

Token Lexer::not_equals() {
	Pos tmp_pos = current_pos;
	current_pos.advance(2);
	return Token(Token::category::not_equals, "!=", tmp_pos);
}

Token Lexer::asterisk() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::asterisk, "*", tmp_pos);
}
Token Lexer::lparen() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::lparen, "(", tmp_pos);
}

Token Lexer::rparen() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::rparen, ")", tmp_pos);
}

Token Lexer::dbl_quoted_string() {
	Pos tmp_pos = current_pos;

	std::string value;
	do {
		if (test_eof() || test_newline()) {
			throw std::runtime_error(std::string(current_pos) + " -> ERROR: expected closing double quote");
		}

		if (test_escaped_character()) {
			value += escaped_character();
			continue;
		}

		value += (*input)[current_pos];
		current_pos.advance();
	} while (!test_dbl_quote());

	value += (*input)[current_pos];
	current_pos.advance(); //advance over closing quote

	return Token(Token::category::dbl_quoted_string, value, tmp_pos);
}


char Lexer::escaped_character() {
	current_pos.advance();

	if (test_eof() || test_newline()) {
		throw std::runtime_error(std::string(current_pos) + " -> ERROR: expected escaped character");
	}

	char res = (*input)[current_pos];
	current_pos.advance();

	return res;
}

Token Lexer::id() {
	Pos tmp_pos = current_pos;
	std::string value;
	size_t shift = 0;

	while ((test_id(shift) || (*input)[current_pos + shift] == '-' ||  isdigit((*input)[current_pos + shift])) && !test_eof()) {
		value += (*input)[current_pos + shift];
		shift++;
	}

	std::transform(value.begin(), value.end(), value.begin(), toupper);

	//check for buildins

	if (value == "SELECT") {
		return select();
	} else if (value == "FROM") {
		return from();
	} else if (value == "WHERE") {
		return where();
	} else if (value == "NOT") {
		return not_();
	} else if (value == "AND") {
		return and_();
	} else if (value == "OR") {
		return or_();
	} else {
		current_pos.advance(shift);
		return Token(Token::category::id, value, tmp_pos);
	}
}

Token Lexer::select() {
	Pos tmp_pos = current_pos;
	std::string value("SELECT");
	current_pos.advance(value.length());
	return Token(Token::category::select, value, tmp_pos);
}

Token Lexer::from() {
	Pos tmp_pos = current_pos;
	std::string value("FROM");
	current_pos.advance(value.length());
	return Token(Token::category::from, value, tmp_pos);
}

Token Lexer::where() {
	Pos tmp_pos = current_pos;
	std::string value("WHERE");
	current_pos.advance(value.length());
	return Token(Token::category::where, value, tmp_pos);
}

Token Lexer::not_() {
	Pos tmp_pos = current_pos;
	std::string value("NOT");
	current_pos.advance(value.length());
	return Token(Token::category::not_, value, tmp_pos);
}

Token Lexer::and_() {
	Pos tmp_pos = current_pos;
	std::string value("AND");
	current_pos.advance(value.length());
	return Token(Token::category::and_, value, tmp_pos);
}

Token Lexer::or_() {
	Pos tmp_pos = current_pos;
	std::string value("OR");
	current_pos.advance(value.length());
	return Token(Token::category::or_, value, tmp_pos);
}

Token Lexer::get_next_token() {
	while (!test_eof()) {
		if (test_number()) {
			return number();
		} else if (test_id()) {
			return id();
		} else if (test_dbl_quote()) {
			return dbl_quoted_string();
		} else if (test_comma()) {
			return comma();
		} else if (test_equals()) {
			return equals();
		} else if (test_not_equals()) {
			return not_equals();
		} else if (test_asterisk()) {
			return asterisk();
		} else if (test_lparen()) {
			return lparen();
		} else if (test_rparen()) {
			return rparen();
		} else if (test_space()) {
			skip_spaces();
			continue;
		} else {
			throw std::runtime_error(std::string(current_pos) + " -> ERROR: Unknown lexem: " + (*input)[current_pos]);
		}
	}

	return Token(Token::category::eof, "", current_pos);
}

}
