
#pragma once
#include "Pos.hpp"
#include "Token.hpp"
#include <string>

namespace screen_selection {

struct Lexer {
	Lexer(const std::string& input): input(input) {}

	Token get_next_token();

private:
	Pos current_pos;
	std::string input;
};

}


