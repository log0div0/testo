
#pragma once

#include "stb/Image.hpp"
#include "Node.hpp"
#include <string>

namespace screen_selection {

std::string text_to_query(const std::string& text);

struct ScreenSelector {
	bool exec(stb::Image& image, const std::string& query);

	std::shared_ptr<AST::SelectStmt> current_select_stmt;
	std::string current_query;
};

}
