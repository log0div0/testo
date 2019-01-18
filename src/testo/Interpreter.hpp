
#pragma once

#include "Parser.hpp"
#include "Environment.hpp"
#include "Register.hpp"
#include "ControllerCreator.hpp"
#include "Utils.hpp"

struct Interpreter {
	Interpreter(Environment& env, ControllerCreator& cc, const fs::path& file);
	int run();
private:
	Environment& env;
	ControllerCreator& cc;
	Parser parser;
	Register reg;
};
