
#include <Lexer.hpp>
#include <fstream>
#include <stdexcept>
#include <string.h>

Lexer::Lexer(const fs::path& file) {
	std::ifstream input_stream(file);

	if (!input_stream) {
		throw std::runtime_error("Can't open file: " + std::string(file));
	}

	input = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
	current_pos = Pos(file, input);
}

void Lexer::skip_spaces() {
	while (test_space() && !test_eof()) {
		current_pos.advance();
	}
}

void Lexer::skip_comments() {
	while (!test_newline() && !test_eof()) {
		current_pos.advance();
	}
}

void Lexer::handle_ifdef() {
	if (is_inside_if) {
		throw std::runtime_error(std::string(current_pos) + ": Error: nested ifdefs are not supported");
	}

	skip_spaces();
	//NEED to get id. Otherwise - error
	if (test_eof() || !test_id()) {
		throw std::runtime_error(std::string(current_pos) + ": Error: ifdef directive has to have an ID");
	}

	is_inside_if = true;
	Token env_var = id();
	if (test_eof()) {
		return;
	}

	if (!test_newline()) {
		throw std::runtime_error(std::string(current_pos) + ": Error: ifdef directive has be finished with a newline");
	}

	current_pos.advance();

	auto env_value = std::getenv(env_var.value().c_str());

	if (env_value && strlen(env_value)) {
		return;
	} else {
		//go right to the #endif shit
		if (test_eof()) {
			return;
		}

		while (!test_ppd()) {
			if (test_eof()) {
				return;
			}
			current_pos.advance();
		}

		Token ppd = get_ppd();
		if (ppd.type() == Token::category::endif) {
			return handle_endif();
		} else if (ppd.type() == Token::category::else_) {
			return handle_else(true);
		} else {
			throw std::runtime_error(std::string(current_pos) + ": expected $endif");
		}
	}
}

void Lexer::handle_else(bool should_happen = false) {
	if (!is_inside_if) {
		throw std::runtime_error(std::string(current_pos) + ": Error: $else directive with no opening ifdef");
	}

	if (test_eof()) {
		return;
	}

	if (!test_newline()) {
		throw std::runtime_error(std::string(current_pos) + ": Error: $else directive has be finished with a newline");
	}

	current_pos.advance(); //newline

	if (!should_happen) {
		if (test_eof()) {
			return;
		}

		while (!test_ppd()) {
			if (test_eof()) {
				return;
			}
			current_pos.advance();
		}

		Token ppd = get_ppd();
		if (ppd.type() == Token::category::endif) {
			return handle_endif();
		} else {
			throw std::runtime_error(std::string(current_pos) + ": expected $endif");
		}
	}
}

void Lexer::handle_endif() {
	if (!is_inside_if) {
		throw std::runtime_error(std::string(current_pos) + ": Error: $endif directive with no opening ifdef");
	}

	is_inside_if = false;

	if (test_eof()) {
		return;
	}

	if (!test_newline()) {
		throw std::runtime_error(std::string(current_pos) + ": Error: $endif directive has to be finished with a newline");
	}

	current_pos.advance(); //newline
}

Token Lexer::get_ppd() {
	Pos tmp_pos = current_pos;
	std::string value;
	size_t shift = 0;

	value += input[current_pos + shift]; //handle dollar sign
	shift++;

	while (!test_eof(shift) && test_id(shift)) {
		value += input[current_pos + shift];
		shift++;
	}

	if (value == "$ifdef") {
		return ifdef();
	} else if (value == "$ifndef") {
		//TODO
	} else if (value == "$else") {
		return else_();
	} else if (value == "$endif") {
		return endif();
	} else if (value == "$include") {
		return include();
	} else {
		throw std::runtime_error(std::string(tmp_pos) + " -> ERROR: Unknown ppd directive: " + value);
	}

	return Token(); //keep compiler happy
}

Token Lexer::handle_ppd() {
	Token ppd = get_ppd();
	if (ppd.type() == Token::category::ifdef) {
		handle_ifdef();
	} else if (ppd.type() == Token::category::else_) {
		handle_else();
	} else if (ppd.type() == Token::category::endif) {
		handle_endif();
	} else if (ppd.type() == Token::category::include) {
		return ppd;
	}

	return Token();
}

bool Lexer::test_size_specifier() const {
	if (test_eof(1)) {
		return false;
	}

	if ((input[current_pos] == 'M') || 
		(input[current_pos] == 'G')) 
	{
		if (input[current_pos + 1] == 'b') {
			if (test_eof(2) || (!test_id(2) && !test_number(2))) {
				return true;
			}
		}
	}

	return false;
}

bool Lexer::test_time_specifier() const {
	if (input[current_pos] == 's' ||
		input[current_pos] == 'h' ||
		input[current_pos] == 'm') 
	{
		if (test_eof(1) || (!test_id(1) && !test_number(1))) {
			return true;
		}
	}

	return false;
}

char Lexer::escaped_character() {
	current_pos.advance();

	if (test_eof() || test_newline()) {
		throw std::runtime_error(std::string(current_pos) + " -> ERROR: expected escaped character");
	}

	char res = input[current_pos];
	current_pos.advance();

	return res;
}

Token Lexer::newline() {
	Pos tmp_pos = current_pos;
	while ((test_newline() || test_space()) && !test_eof() ) {
		current_pos.advance();
	}

	return Token(Token::category::newline, "\n", tmp_pos);
}

Token Lexer::number() {
	Pos tmp_pos = current_pos;
	std::string value;

	while (test_number() && !test_eof()) {
		value += input[current_pos];
		current_pos.advance();
	}

	if (test_eof()) {
		return Token(Token::category::number, value, tmp_pos);
	}

	if (test_size_specifier()) {
		return size(value, current_pos);
	} else if (test_time_specifier()) {
		return time_interval(value, current_pos);
	} else if (test_id()) {
		throw std::runtime_error(std::string(tmp_pos) + " -> ERROR: ID can't start with a number");
	} else {
		return Token(Token::category::number, value, tmp_pos);
	}

}

Token Lexer::time_interval(std::string time_number, const Pos& time_number_pos) {
	time_number += input[current_pos];
	current_pos.advance();
	return Token(Token::category::time_interval, time_number, time_number_pos);
}

Token Lexer::size(std::string size_number, const Pos& size_number_pos) {
	size_number += input[current_pos];
	current_pos.advance();
	size_number += input[current_pos];
	current_pos.advance();
	return Token(Token::category::size, size_number, size_number_pos);
}

Token Lexer::id() {
	Pos tmp_pos = current_pos;
	std::string value;
	size_t shift = 0;

	while ((test_id(shift) || isdigit(input[current_pos + shift])) && !test_eof()) {
		value += input[current_pos + shift];
		shift++;
	}

	//check for buildins

	if (value == "type") {
		return type();
	} else if (value == "wait") {
		return wait();
	} else if (value == "press") {
		return press();
	} else if (value == "plug") {
		return plug();
	} else if (value == "unplug") {
		return unplug();
	} else if (value == "start") {
		return start();
	} else if (value == "stop") {
		return stop();
	} else if (value == "exec") {
		return exec();
	} else if (value == "set") {
		return set();
	} else if (value == "copyto") {
		return copyto();
	} else if (value == "for") {
		return for_();
	} else if (value == "snapshot") {
		return snapshot();
	} else if (value == "test") {
		return test();
	} else if (value == "machine") {
		return machine();
	} else if (value == "flash") {
		return flash();
	} else if (value == "dvd") {
		return dvd();
	} else {
		current_pos.advance(shift);
		return Token(Token::category::id, value, tmp_pos);
	}
}

Token Lexer::type() {
	Pos tmp_pos = current_pos;
	std::string value("type");
	current_pos.advance(value.length());
	return Token(Token::category::type_, value, tmp_pos);
}

Token Lexer::wait() {
	Pos tmp_pos = current_pos;
	std::string value("wait");
	current_pos.advance(value.length());
	return Token(Token::category::wait, value, tmp_pos);
}

Token Lexer::press() {
	Pos tmp_pos = current_pos;
	std::string value("press");
	current_pos.advance(value.length());
	return Token(Token::category::press, value, tmp_pos);
}

Token Lexer::plug() {
	Pos tmp_pos = current_pos;
	std::string value("plug");
	current_pos.advance(value.length());
	return Token(Token::category::plug, value, tmp_pos);
}

Token Lexer::unplug() {
	Pos tmp_pos = current_pos;
	std::string value("unplug");
	current_pos.advance(value.length());
	return Token(Token::category::unplug, value, tmp_pos);
}

Token Lexer::start() {
	Pos tmp_pos = current_pos;
	std::string value("start");
	current_pos.advance(value.length());
	return Token(Token::category::start, value, tmp_pos);
}

Token Lexer::stop() {
	Pos tmp_pos = current_pos;
	std::string value("stop");
	current_pos.advance(value.length());
	return Token(Token::category::stop, value, tmp_pos);
}

Token Lexer::exec() {
	Pos tmp_pos = current_pos;
	std::string value("exec");
	current_pos.advance(value.length());
	return Token(Token::category::exec, value, tmp_pos);
}

Token Lexer::set() {
	Pos tmp_pos = current_pos;
	std::string value("set");
	current_pos.advance(value.length());
	return Token(Token::category::set, value, tmp_pos);
}

Token Lexer::copyto() {
	Pos tmp_pos = current_pos;
	std::string value("copyto");
	current_pos.advance(value.length());
	return Token(Token::category::copyto, value, tmp_pos);
}

Token Lexer::for_() {
	Pos tmp_pos = current_pos;
	std::string value("for");
	current_pos.advance(value.length());
	return Token(Token::category::for_, value, tmp_pos);
}

Token Lexer::snapshot() {
	Pos tmp_pos = current_pos;
	std::string value("snapshot");
	current_pos.advance(value.length());
	return Token(Token::category::snapshot, value, tmp_pos);
}

Token Lexer::test() {
	Pos tmp_pos = current_pos;
	std::string value("test");
	current_pos.advance(value.length());
	return Token(Token::category::test, value, tmp_pos);
}

Token Lexer::machine() {
	Pos tmp_pos = current_pos;
	std::string value("machine");
	current_pos.advance(value.length());
	return Token(Token::category::machine, value, tmp_pos);
}

Token Lexer::flash() {
	Pos tmp_pos = current_pos;
	std::string value("flash");
	current_pos.advance(value.length());
	return Token(Token::category::flash, value, tmp_pos);
}

Token Lexer::dvd() {
	Pos tmp_pos = current_pos;
	std::string value("dvd");
	current_pos.advance(value.length());
	return Token(Token::category::dvd, value, tmp_pos);
}

Token Lexer::multiline_string() {
	Pos tmp_pos = current_pos;

	std::string value;

	//advance over first 3 characters for we already know that it's a triple quote
	//If we don't do so, we could trip over something like """"
	value += input[current_pos];
	current_pos.advance();
	value += input[current_pos];
	current_pos.advance();
	value += input[current_pos];
	current_pos.advance();

	do {
		if (test_eof()) {
			throw std::runtime_error(std::string(current_pos) + " -> ERROR: expected closing triple quote");
		}

		value += input[current_pos];
		current_pos.advance();
	} while (!test_multiline_quote());

	value += input[current_pos];
	current_pos.advance(); //advance over closing quote
	value += input[current_pos];
	current_pos.advance();
	value += input[current_pos];
	current_pos.advance();

	return Token(Token::category::multiline_string, value, tmp_pos);
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
		}

		value += input[current_pos];
		current_pos.advance();
	} while (!test_dbl_quote());

	value += input[current_pos];
	current_pos.advance(); //advance over closing quote

	return Token(Token::category::dbl_quoted_string, value, tmp_pos);
}

Token Lexer::comma() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::comma, ",", tmp_pos);
}

Token Lexer::assign() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::assign, "=", tmp_pos);
}

Token Lexer::plus() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::plus, "+", tmp_pos);
}

Token Lexer::asterisk() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::asterisk, "*", tmp_pos);
}

Token Lexer::lbrace() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::lbrace, "{", tmp_pos);
}

Token Lexer::rbrace() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::rbrace, "}", tmp_pos);
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

Token Lexer::semi() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::semi, ";", tmp_pos);
}

Token Lexer::colon() {
	Pos tmp_pos = current_pos;
	current_pos.advance();
	return Token(Token::category::colon, ":", tmp_pos);
}

Token Lexer::ifdef() {
	Pos tmp_pos = current_pos;
	std::string value("$ifdef");
	current_pos.advance(value.length());
	return Token(Token::category::ifdef, value, tmp_pos);
}

Token Lexer::endif() {
	Pos tmp_pos = current_pos;
	std::string value("$endif");
	current_pos.advance(value.length());
	return Token(Token::category::endif, value, tmp_pos);
}

Token Lexer::else_() {
	Pos tmp_pos = current_pos;
	std::string value("$else");
	current_pos.advance(value.length());
	return Token(Token::category::else_, value, tmp_pos);
}

Token Lexer::include() {
	Pos tmp_pos = current_pos;
	std::string value("$include");
	current_pos.advance(value.length());
	return Token(Token::category::include, value, tmp_pos);
}

Token Lexer::get_next_token() {
	while (!test_eof()) {
		if (test_newline()) {
			return newline();
		} else if (test_number()) {
			return number();
		} else if (test_id()) {
			return id();
		} else if (test_multiline_quote()) {
			return multiline_string();
		} else if (test_dbl_quote()) {
			return dbl_quoted_string();
		} else if (test_comma()) {
			return comma();
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
		} else if (test_semi()) {
			return semi();
		} else if (test_colon()) {
			return colon();
		} else if (test_ppd()) {
			auto res = handle_ppd();
			if (res.type() == Token::category::include) {
				return res;
			} 
			continue;
		} else if (test_space()) {
			skip_spaces();
			continue;
		} else if (test_comments()) {
			skip_comments();
			continue;
		} else {
			throw std::runtime_error(std::string(current_pos) + " -> ERROR: Unknown lexem: " + input[current_pos]);
		}
	}

	if (is_inside_if) {
		throw std::runtime_error("Reached end of file in $if(n)def without $endif ppd");
	}
	return Token(Token::category::eof, "", current_pos);
}

