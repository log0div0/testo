
#pragma once
#include <Pos.hpp>
#include <Token.hpp>
#include <Utils.hpp>
#include <string>
#include <unordered_map>
#include <functional>

struct Lexer {
	Lexer(const fs::path& file);

	Token get_next_token();
	fs::path file() const {
		return current_pos.file;
	}

private:

	bool test_eof(size_t shift = 0) const { return ((current_pos + shift) == input.length()); }
	bool test_newline() const { return (input[current_pos] == '\n'); }
	bool test_number(size_t shift = 0) const { return isdigit(input[current_pos + shift]); }
	bool test_id(size_t shift = 0) const {
		return (isalpha(input[current_pos + shift]) || 
			(input[current_pos + shift] == '_'));
	}
	bool test_comments() const { return (input[current_pos] == '#'); }
	bool test_begin_multiline_comments() const {
		if (test_eof(1)) {
			return false;
		}
		return ((input[current_pos] == '/') &&
			(input[current_pos + 1] == '*'));
	}

	bool test_end_multiline_comments() const {
		if (test_eof(1)) {
			return false;
		}
		return ((input[current_pos] == '*') &&
			(input[current_pos + 1] == '/'));
	}
	bool test_space(size_t shift = 0) const {
		char c = input[current_pos + shift];

		return ((c == ' ') ||
			(c == '\r') ||
			(c == '\t'));
	}
	bool test_dbl_quote() const { return (input[current_pos] == '\"'); }
	bool test_multiline_quote() const {
		if (test_eof(1)) {
			return false;
		} else if (test_eof(2)) {
			return false;
		}

		char quote = '\"';
		return ((input[current_pos] == quote) &&
			(input[current_pos + 1] == quote) &&
			(input[current_pos + 2] == quote));
	}
	bool test_assign() const { return input[current_pos] == '='; }
	bool test_time_specifier() const;
	bool test_size_specifier() const;

	bool test_escaped_character() const { return  (input[current_pos] == '\\'); }

	bool test_comma() const { return (input[current_pos] == ','); }
	bool test_plus() const { return (input[current_pos] == '+'); }
	bool test_asterisk() const { return (input[current_pos] == '*'); }
	bool test_lbrace() const { return (input[current_pos] == '{'); }
	bool test_rbrace() const { return (input[current_pos] == '}'); }
	bool test_lparen() const { return (input[current_pos] == '('); }
	bool test_rparen() const { return (input[current_pos] == ')'); }
	bool test_semi() const { return (input[current_pos] == ';'); }
	bool test_colon() const { return (input[current_pos] == ':'); }
	bool test_ppd() const { return (input[current_pos] == '$' && current_pos.column == 1); }

	void skip_spaces();
	void skip_comments();
	void skip_multiline_comments();

	Token handle_ppd(); //preprocessor directive
	void handle_ifdef();
	void handle_else(bool should_happen);
	void handle_endif();

	char escaped_character();

	Token newline();
	Token number();
	Token time_interval(std::string time_number, const Pos& time_number_pos);
	Token size(std::string size_number, const Pos& size_number_pos);
	Token id();
	Token type();
	Token wait();
	Token press();
	Token plug();
	Token unplug();
	Token start();
	Token stop();
	Token exec();
	Token set();
	Token copyto();
	Token for_();
	Token snapshot();
	Token test();
	Token machine();
	Token flash();
	Token macro();
	Token dvd();
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
	Token semi();
	Token colon();
	Token get_ppd();
	Token ifdef();
	Token endif();
	Token else_();
	Token include();

	Pos current_pos;
	std::string input;
	bool is_inside_if = false;
};