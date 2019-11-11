
#include "ScreenSelector.hpp"
#include "Parser.hpp"
#include "VisitorSemantic.hpp"
#include "VisitorInterpreter.hpp"

namespace screen_selection {

std::string text_to_query(const std::string& text) {
	return "";
}

bool ScreenSelector::exec(stb::Image& image, const std::string& query) {
	//1) Tokenize the whole shit
	//2) Parse

	Parser parser(query);
	auto select_stmt = parser.parse();
	VisitorSemantic semantic;
	semantic.visit(select_stmt);
	VisitorInterpreter runner;
	return runner.visit(select_stmt);
}

}