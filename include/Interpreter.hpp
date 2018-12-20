
#pragma once

#include <Parser.hpp>
#include <Global.hpp>
#include <Utils.hpp>

struct Interpreter {
	Interpreter(const fs::path& file);
	int run();
	Global global;
private:
	Parser parser;
};
