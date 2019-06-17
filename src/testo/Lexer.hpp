
#pragma once
#include "Pos.hpp"
#include "Token.hpp"
#include "Utils.hpp"
#include <string>
#include <unordered_map>
#include <functional>

struct Lexer {
	Lexer(const fs::path& file, const std::string& input);

	Token get_next_token();
	fs::path file() const {
		return current_pos.file;
	}

private:

	bool test_eof(size_t shift = 0) const { return ((current_pos + shift) == input->length()); }
	bool test_newline() const { return ((*input)[current_pos] == '\n'); }
	bool test_number(size_t shift = 0) const { return isdigit((*input)[current_pos + shift]); }
	bool test_id(size_t shift = 0) const {
		return (isalpha((*input)[current_pos + shift]) ||
			((*input)[current_pos + shift] == '_'));
	}

	bool test_var_ref() const {
		return ((*input)[current_pos] == '$');
	}
	bool test_comments() const { return ((*input)[current_pos] == '#'); }
	bool test_begin_multiline_comments() const {
		if (test_eof(1)) {
			return false;
		}
		return (((*input)[current_pos] == '/') &&
			((*input)[current_pos + 1] == '*'));
	}

	bool test_end_multiline_comments() const {
		if (test_eof(1)) {
			return false;
		}
		return (((*input)[current_pos] == '*') &&
			((*input)[current_pos + 1] == '/'));
	}
	bool test_space(size_t shift = 0) const {
		char c = (*input)[current_pos + shift];

		return ((c == ' ') ||
			(c == '\r') ||
			(c == '\t'));
	}
	bool test_dbl_quote() const { return ((*input)[current_pos] == '\"'); }
	bool test_multiline_quote() const {
		if (test_eof(1)) {
			return false;
		} else if (test_eof(2)) {
			return false;
		}

		char quote = '\"';
		return (((*input)[current_pos] == quote) &&
			((*input)[current_pos + 1] == quote) &&
			((*input)[current_pos + 2] == quote));
	}
	bool test_assign() const { return (*input)[current_pos] == '='; }
	bool test_time_specifier() const;
	bool test_size_specifier() const;

	bool test_escaped_character() const { return  ((*input)[current_pos] == '\\'); }

	bool test_comma() const { return ((*input)[current_pos] == ','); }
	bool test_plus() const { return ((*input)[current_pos] == '+'); }
	bool test_asterisk() const { return ((*input)[current_pos] == '*'); }
	bool test_lbrace() const { return ((*input)[current_pos] == '{'); }
	bool test_rbrace() const { return ((*input)[current_pos] == '}'); }
	bool test_lparen() const { return ((*input)[current_pos] == '('); }
	bool test_rparen() const { return ((*input)[current_pos] == ')'); }
	bool test_lbracket() const { return ((*input)[current_pos] == '['); }
	bool test_rbracket() const { return ((*input)[current_pos] == ']'); }
	bool test_semi() const { return ((*input)[current_pos] == ';'); }
	bool test_colon() const { return ((*input)[current_pos] == ':'); }
	bool test_double_dot() const {
		if (test_eof(1)) {
			return false;
		}
		return (((*input)[current_pos] == '.') &&
			((*input)[current_pos + 1] == '.'));
	}

	void skip_spaces();
	void skip_comments();
	void skip_multiline_comments();

	char escaped_character();

	Token newline();
	Token number();
	Token time_interval(std::string time_number, const Pos& time_number_pos);
	Token size(std::string size_number, const Pos& size_number_pos);
	Token id();
	Token var_ref();
	Token type();
	Token wait();
	Token check();
	Token press();
	Token plug();
	Token unplug();
	Token start();
	Token stop();
	Token shutdown();
	Token exec();
	Token set();
	Token copyto();
	Token copyfrom();
	Token timeout();
	Token for_();
	Token test();
	Token machine();
	Token flash();
	Token macro();
	Token dvd();
	Token if_();
	Token else_();
	Token in();
	Token break_();
	Token continue_();
	Token dbl_quoted_string();
	Token multiline_string();
	Token assign();
	Token comma();
	Token plus();
	Token asterisk();
	Token lbrace();
	Token rbrace();
	Token lparen();
	Token rparen();
	Token lbracket();
	Token rbracket();
	Token semi();
	Token colon();
	Token double_dot();
	Token include();

	//expressions

	Token LESS();
	Token GREATER();
	Token EQUAL();
	Token STRLESS();
	Token STRGREATER();
	Token STREQUAL();
	Token NOT();
	Token AND();
	Token OR();

	Pos current_pos;
	std::shared_ptr<std::string> input;
};
