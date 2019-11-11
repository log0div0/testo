
#pragma once

#include "stb/Image.hpp"
#include "Node.hpp"
#include <string>

namespace tql {

std::string text_to_query(const std::string& text);

struct Interpreter {
	Interpreter(const std::string& query);

	bool exec(stb::Image& image);

	std::shared_ptr<AST::SelectStmt> select_stmt;
};

}
