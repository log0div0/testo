
#include "ScreenSelector.hpp"
#include "Parser.hpp"

namespace screen_selection {

std::string text_to_query(const std::string& text) {
	return "";
}

bool ScreenSelector::exec(stb::Image& image, const std::string& query) {
	//1) Tokenize the whole shit
	//2) Parse

	Parser parser(query);
	//VisitorInterpreter runner;
	auto select_stmt = parser.parse();
	//return runner.run(exec_plan);
	return true;
}

}