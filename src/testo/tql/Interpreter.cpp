
#include "Interpreter.hpp"
#include "Parser.hpp"
#include "VisitorSemantic.hpp"
#include "VisitorInterpreter.hpp"
 #include <fmt/format.h>

namespace tql {

std::string text_to_query(const std::string& text) {
	std::string tmp_str;
	for (auto& c: text) {
		if (c == '\"') {
			tmp_str += "\\";
		}
		tmp_str += c;
	}
	std::string result = fmt::format("select * from textlines where text = \"{}\"", tmp_str);
	return result;
}

Interpreter::Interpreter(const std::string& query): input(query) {
	Parser parser(query);
	select_stmt = parser.parse();
	VisitorSemantic semantic;
	semantic.visit(select_stmt);
}

bool Interpreter::exec(stb::Image& image) {
	VisitorInterpreter runner(image);
	return runner.visit(select_stmt);
}

void Interpreter::validate_sanity(const std::string& query) {
	Interpreter interpreter(query);
}

}
