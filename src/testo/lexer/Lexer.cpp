
#include "Lexer.hpp"
#include <fstream>
#include <stdexcept>
#include <string.h>
#include "../Exceptions.hpp"
#include "../Logger.hpp"

Lexer::Lexer(const fs::path& file, const std::string& input): input(new std::string(input)) {
	TRACE();
	current_pos = Pos(file, this->input);
}

Lexer::~Lexer() {
	TRACE();
}

void Lexer::advance(size_t shift) {
	current_pos.advance(shift - 1);
	previous_pos = current_pos;
	current_pos.advance();
}

void Lexer::skip_spaces() {
	while (test_space() && !test_eof()) {
		advance();
	}
}

void Lexer::skip_comments() {
	while (!test_newline() && !test_eof()) {
		advance();
	}
}

void Lexer::skip_multiline_comments() {
	while(!test_end_multiline_comments()) {
		advance();
		if (test_eof()) {
			throw ExceptionWithPos(current_pos, "Error: can't find end of multiline comments");
		}

		if (test_begin_multiline_comments()) {
			throw ExceptionWithPos(current_pos, "Error: nested multiline comments are not allowed");
		}
	}

	advance();
	advance();
}


bool Lexer::test_size_specifier() const {
	if (test_eof(1)) {
		return false;
	}

	if (((*input)[current_pos] == 'K') ||
		((*input)[current_pos] == 'M') ||
		((*input)[current_pos] == 'G'))
	{
		if ((*input)[current_pos + 1] == 'b') {
			if (test_eof(2) || (!test_id(2) && !test_digit(2))) {
				return true;
			}
		}
	}

	return false;
}

bool Lexer::test_time_specifier() const {
	if ((*input)[current_pos] == 's' ||
		(*input)[current_pos] == 'h')
	{
		if (test_eof(1) || (!test_id(1) && !test_digit(1))) {
			return true;
		}
	}

	if ((*input)[current_pos] == 'm') {
		if (test_eof(1)) {
			return true;
		}
		if ((*input)[current_pos + 1] == 's') {
			if (test_eof(2) || (!test_id(2) && !test_digit(2))) {
				return true;
			}
		} else {
			if (test_eof(1) || (!test_id(1) && !test_digit(1))) {
				return true;
			}
		}
	}

	return false;
}

char Lexer::escaped_character() {
	advance();

	if (!test_eof() && test_newline()) {
		advance();
	}

	if (test_eof()) {
		throw ExceptionWithPos(current_pos, "Error: Unexpected eof");
	}

	char res = (*input)[current_pos];
	advance();

	return res;
}

Token Lexer::newline() {
	Pos tmp_pos = current_pos;
	while ((test_newline() || test_space()) && !test_eof() ) {
		advance();
	}

	return Token(Token::category::newline, "\n", tmp_pos, tmp_pos);
}

Token Lexer::number() {
	Pos tmp_pos = current_pos;

	std::string value;

	bool is_signed = false;

	if (test_plus() || test_minus()) {
		is_signed = true;
		value += (*input)[current_pos];
		advance();
	}

	while (test_digit() && !test_eof()) {
		value += (*input)[current_pos];
		advance();
	}

	if (test_eof()) {
		return Token(Token::category::number, value, tmp_pos, previous_pos);
	}

	if (test_size_specifier()) {
		if (is_signed) {
			throw ExceptionWithPos(tmp_pos, "Error: size specifier can't be signed");
		}
		return size(value, current_pos);
	} else if (test_time_specifier()) {
		if (is_signed) {
			throw ExceptionWithPos(tmp_pos, "Error: time specifier can't be signed");
		}
		return time_interval(value, current_pos);
	} else if (test_id()) {
		throw ExceptionWithPos(tmp_pos, "Error: ID can't start with a number");
	} else {
		return Token(Token::category::number, value, tmp_pos, previous_pos);
	}

}

Token Lexer::time_interval(std::string time_number, const Pos& time_number_pos) {
	time_number += (*input)[current_pos];
	advance();
	if ((*input)[current_pos] == 's') {
		time_number += (*input)[current_pos];
		advance();
	}
	return Token(Token::category::time_interval, time_number, time_number_pos, previous_pos);
}

Token Lexer::size(std::string size_number, const Pos& size_number_pos) {
	size_number += (*input)[current_pos];
	advance();
	size_number += (*input)[current_pos];
	advance();
	return Token(Token::category::size, size_number, size_number_pos, previous_pos);
}

Token Lexer::id() {
	Pos tmp_pos = current_pos;
	std::string value;
	size_t shift = 0;

	while ((test_id(shift) || (*input)[current_pos + shift] == '-' ||  isdigit((*input)[current_pos + shift])) && !test_eof()) {
		value += (*input)[current_pos + shift];
		shift++;
	}

	//check for buildins

	if (value == "for") {
		return for_();
	} else if (value == "test") {
		return test();
	} else if (value == "machine") {
		return machine();
	} else if (value == "flash") {
		return flash();
	} else if (value == "network") {
		return network();
	} else if (value == "param") {
		return param();
	} else if (value == "macro") {
		return macro();
	} else if (value == "if") {
		return if_();
	} else if (value == "else") {
		return else_();
	} else if (value == "IN") {
		return IN_();
	} else if (value == "RANGE") {
		return RANGE();
	} else if (value == "break") {
		return break_();
	} else if (value == "continue") {
		return continue_();
	} else if (value == "include") {
		return include();
	} else if (value == "true") {
		return boolean("true");
	} else if (value == "false") {
		return boolean("false");
	} else if (value == "DEFINED") {
		return DEFINED();
	} else if (value == "LESS") {
		return LESS();
	} else if (value == "GREATER") {
		return GREATER();
	} else if (value == "EQUAL") {
		return EQUAL();
	}  else if (value == "STRLESS") {
		return STRLESS();
	}  else if (value == "STRGREATER") {
		return STRGREATER();
	}  else if (value == "STREQUAL") {
		return STREQUAL();
	}  else if (value == "NOT") {
		return NOT();
	}  else if (value == "AND") {
		return AND();
	}  else if (value == "OR") {
		return OR();
	} else {
		advance(shift);
		return Token(Token::category::id, value, tmp_pos, previous_pos);
	}
}

Token Lexer::for_() {
	Pos tmp_pos = current_pos;
	std::string value("for");
	advance(value.length());
	return Token(Token::category::for_, value, tmp_pos, previous_pos);
}

Token Lexer::test() {
	Pos tmp_pos = current_pos;
	std::string value("test");
	advance(value.length());
	return Token(Token::category::test, value, tmp_pos, previous_pos);
}

Token Lexer::machine() {
	Pos tmp_pos = current_pos;
	std::string value("machine");
	advance(value.length());
	return Token(Token::category::machine, value, tmp_pos, previous_pos);
}

Token Lexer::flash() {
	Pos tmp_pos = current_pos;
	std::string value("flash");
	advance(value.length());
	return Token(Token::category::flash, value, tmp_pos, previous_pos);
}

Token Lexer::network() {
	Pos tmp_pos = current_pos;
	std::string value("network");
	advance(value.length());
	return Token(Token::category::network, value, tmp_pos, previous_pos);
}

Token Lexer::param() {
	Pos tmp_pos = current_pos;
	std::string value("param");
	advance(value.length());
	return Token(Token::category::param, value, tmp_pos, previous_pos);
}

Token Lexer::macro() {
	Pos tmp_pos = current_pos;
	std::string value("macro");
	advance(value.length());
	return Token(Token::category::macro, value, tmp_pos, previous_pos);
}

Token Lexer::if_() {
	Pos tmp_pos = current_pos;
	std::string value("if");
	advance(value.length());
	return Token(Token::category::if_, value, tmp_pos, previous_pos);
}

Token Lexer::else_() {
	Pos tmp_pos = current_pos;
	std::string value("else");
	advance(value.length());
	return Token(Token::category::else_, value, tmp_pos, previous_pos);
}

Token Lexer::IN_() {
	Pos tmp_pos = current_pos;
	std::string value("IN");
	advance(value.length());
	return Token(Token::category::IN_, value, tmp_pos, previous_pos);
}

Token Lexer::RANGE() {
	Pos tmp_pos = current_pos;
	std::string value("RANGE");
	advance(value.length());
	return Token(Token::category::RANGE, value, tmp_pos, previous_pos);
}

Token Lexer::break_() {
	Pos tmp_pos = current_pos;
	std::string value("break");
	advance(value.length());
	return Token(Token::category::break_, value, tmp_pos, previous_pos);
}

Token Lexer::continue_() {
	Pos tmp_pos = current_pos;
	std::string value("continue");
	advance(value.length());
	return Token(Token::category::continue_, value, tmp_pos, previous_pos);
}

Token Lexer::include() {
	Pos tmp_pos = current_pos;
	std::string value("include");
	advance(value.length());
	return Token(Token::category::include, value, tmp_pos, previous_pos);
}

Token Lexer::boolean(const std::string& value) {
	Pos tmp_pos = current_pos;
	advance(value.length());
	return Token(Token::category::boolean, value, tmp_pos, previous_pos);
}

Token Lexer::DEFINED() {
	Pos tmp_pos = current_pos;
	std::string value("DEFINED");
	advance(value.length());
	return Token(Token::category::DEFINED, value, tmp_pos, previous_pos);
}

Token Lexer::LESS() {
	Pos tmp_pos = current_pos;
	std::string value("LESS");
	advance(value.length());
	return Token(Token::category::LESS, value, tmp_pos, previous_pos);
}

Token Lexer::GREATER() {
	Pos tmp_pos = current_pos;
	std::string value("GREATER");
	advance(value.length());
	return Token(Token::category::GREATER, value, tmp_pos, previous_pos);
}

Token Lexer::EQUAL() {
	Pos tmp_pos = current_pos;
	std::string value("EQUAL");
	advance(value.length());
	return Token(Token::category::EQUAL, value, tmp_pos, previous_pos);
}

Token Lexer::STRLESS() {
	Pos tmp_pos = current_pos;
	std::string value("STRLESS");
	advance(value.length());
	return Token(Token::category::STRLESS, value, tmp_pos, previous_pos);
}

Token Lexer::STRGREATER() {
	Pos tmp_pos = current_pos;
	std::string value("STRGREATER");
	advance(value.length());
	return Token(Token::category::STRGREATER, value, tmp_pos, previous_pos);
}

Token Lexer::STREQUAL() {
	Pos tmp_pos = current_pos;
	std::string value("STREQUAL");
	advance(value.length());
	return Token(Token::category::STREQUAL, value, tmp_pos, previous_pos);
}

Token Lexer::NOT() {
	Pos tmp_pos = current_pos;
	std::string value("NOT");
	advance(value.length());
	return Token(Token::category::NOT, value, tmp_pos, previous_pos);
}

Token Lexer::AND() {
	Pos tmp_pos = current_pos;
	std::string value("AND");
	advance(value.length());
	return Token(Token::category::AND, value, tmp_pos, previous_pos);
}

Token Lexer::OR() {
	Pos tmp_pos = current_pos;
	std::string value("OR");
	advance(value.length());
	return Token(Token::category::OR, value, tmp_pos, previous_pos);
}

Token Lexer::triple_quoted_string() {
	Pos tmp_pos = current_pos;

	std::string value;

	//advance over first 3 characters for we already know that it's a triple quote
	//If we don't do so, we could trip over something like """"
	value += (*input)[current_pos];
	advance();
	value += (*input)[current_pos];
	advance();
	value += (*input)[current_pos];
	advance();

	do {
		if (test_eof()) {
			throw ExceptionWithPos(current_pos, "Error: expected closing triple quote");
		}

		if (test_escaped_character()) {
			value += escaped_character();
			continue;
		}

		value += (*input)[current_pos];
		advance();
	} while (!test_triple_quote());

	//Check if we have another quotes over the end. Like """"

	while(test_triple_quote()) {
		value += (*input)[current_pos];
		advance(); //advance over closing quote
	}

	value += (*input)[current_pos];
	advance();
	value += (*input)[current_pos];
	advance();

	return Token(Token::category::triple_quoted_string, value, tmp_pos, previous_pos);
}


Token Lexer::quoted_string() {
	Pos tmp_pos = current_pos;

	std::string value;
	do {
		if (test_eof() || test_newline()) {
			throw ExceptionWithPos(current_pos, "Error: expected closing double quote");
		}

		if (test_escaped_character()) {
			value += escaped_character();
			continue;
		}

		value += (*input)[current_pos];
		advance();
	} while (!test_quote());

	value += (*input)[current_pos];
	advance(); //advance over closing quote

	return Token(Token::category::quoted_string, value, tmp_pos, previous_pos);
}

Token Lexer::dot() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::dot, ".", tmp_pos, previous_pos);
}

Token Lexer::comma() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::comma, ",", tmp_pos, previous_pos);
}

Token Lexer::exclamation_mark() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::exclamation_mark, "!", tmp_pos, tmp_pos);
}

Token Lexer::double_ampersand() {
	Pos tmp_pos = current_pos;
	advance(2);
	return Token(Token::category::double_ampersand, "&&", tmp_pos, previous_pos);
}

Token Lexer::double_vertical_bar() {
	Pos tmp_pos = current_pos;
	advance(2);
	return Token(Token::category::double_vertical_bar, "||", tmp_pos, previous_pos);
}

Token Lexer::assign() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::assign, "=", tmp_pos, tmp_pos);
}

Token Lexer::plus() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::plus, "+", tmp_pos, tmp_pos);
}

Token Lexer::asterisk() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::asterisk, "*", tmp_pos, tmp_pos);
}

Token Lexer::lbrace() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::lbrace, "{", tmp_pos, tmp_pos);
}

Token Lexer::rbrace() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::rbrace, "}", tmp_pos, tmp_pos);
}

Token Lexer::lparen() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::lparen, "(", tmp_pos, tmp_pos);
}

Token Lexer::rparen() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::rparen, ")", tmp_pos, tmp_pos);
}

Token Lexer::lbracket() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::lbracket, "[", tmp_pos, tmp_pos);
}

Token Lexer::rbracket() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::rbracket, "]", tmp_pos, tmp_pos);
}

Token Lexer::semi() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::semi, ";", tmp_pos, tmp_pos);
}

Token Lexer::colon() {
	Pos tmp_pos = current_pos;
	advance();
	return Token(Token::category::colon, ":", tmp_pos, tmp_pos);
}

Token Lexer::get_next_token() {
	while (!test_eof()) {
		if (test_newline()) {
			return newline();
		} else if (test_number()) {
			return number();
		} else if (test_id()) {
			return id();
		} else if (test_triple_quote()) {
			return triple_quoted_string();
		} else if (test_quote()) {
			return quoted_string();
		} else if (test_dot()) {
			return dot();
		} else if (test_comma()) {
			return comma();
		} else if (test_exclamation_mark()) {
			return exclamation_mark();
		} else if (test_double_ampersand()) {
			return double_ampersand();
		} else if (test_double_vertical_bar()) {
			return double_vertical_bar();
		} else if (test_assign()) {
			return assign();
		} else if (test_plus()) {
			return plus();
		} else if (test_asterisk()) {
			return asterisk();
		} else if (test_lbrace()) {
			return lbrace();
		} else if (test_rbrace()) {
			return rbrace();
		} else if (test_lparen()) {
			return lparen();
		} else if (test_rparen()) {
			return rparen();
		} else if (test_lbracket()) {
			return lbracket();
		} else if (test_rbracket()) {
			return rbracket();
		} else if (test_semi()) {
			return semi();
		} else if (test_colon()) {
			return colon();
		} else if (test_space()) {
			skip_spaces();
			continue;
		} else if (test_comments()) {
			skip_comments();
			continue;
		} else if (test_begin_multiline_comments()) {
			skip_multiline_comments();
			continue;
		} else {
			throw ExceptionWithPos(current_pos, "Error: Unknown lexem: " + (*input)[current_pos]);
		}
	}

	return Token(Token::category::eof, "", current_pos, current_pos);
}
