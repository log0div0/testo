
#pragma once

#include "stb/Image.hpp"
#include "Node.hpp"
#include <string>

namespace tql {

std::string text_to_query(const std::string& text);

struct Interpreter {
	Interpreter(const std::string& query);

	bool exec(stb::Image& image);
	static void validate_sanity(const std::string& query);

	std::shared_ptr<AST::SelectStmt> select_stmt;
	std::string input;
};

}
