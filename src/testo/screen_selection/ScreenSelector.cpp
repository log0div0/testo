
#include "ScreenSelector.hpp"
#include "Parser.hpp"
#include "VisitorSemantic.hpp"
#include "VisitorInterpreter.hpp"
 #include <fmt/format.h>

namespace screen_selection {

std::string text_to_query(const std::string& text) {
	std::string result = fmt::format("select * from textlines where text = {}", text);
	return result;
}

bool ScreenSelector::exec(stb::Image& image, const std::string& query) {
	//1) Tokenize the whole shit
	//2) Parse

	Parser parser(query);
	auto select_stmt = parser.parse();
	VisitorSemantic semantic;
	semantic.visit(select_stmt);
	VisitorInterpreter runner(image);
	return runner.visit(select_stmt);
}

}