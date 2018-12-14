
#pragma once

#include <Parser.hpp>
#include <Global.hpp>

struct Interpreter {
	Interpreter(const std::string& file);
	int run();
	Global global;
private:
	Parser parser;
};
