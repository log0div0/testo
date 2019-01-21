
#pragma once

#include "Parser.hpp"
#include "Environment.hpp"
#include "Register.hpp"
#include "Utils.hpp"

struct Interpreter {
	Interpreter(Environment& env, const fs::path& file);
	int run();
private:
	Environment& env;
	Parser parser;
	Register reg;
};
