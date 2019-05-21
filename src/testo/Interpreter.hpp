
#pragma once

#include "Parser.hpp"
#include "backends/Environment.hpp"
#include "Register.hpp"
#include "Utils.hpp"

struct Interpreter {
	Interpreter(Environment& env, const fs::path& file, const nlohmann::json& config);
	Interpreter(Environment& env, const fs::path& dir, const std::string& input, const nlohmann::json& config);
	int run();
private:
	Environment& env;
	Parser parser;
	nlohmann::json config;
	Register reg;
};
