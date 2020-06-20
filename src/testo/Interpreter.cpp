
#include "Interpreter.hpp"
#include "VisitorSemantic.hpp"
#include "VisitorInterpreter.hpp"
#include <fstream>

Interpreter::Interpreter(const fs::path& file, const nlohmann::json& config):
	reg(new Register()),
	config(config)
{
	std::ifstream input_stream(file);

	if (!input_stream) {
		throw std::runtime_error("Can't open file: " + file.generic_string());
	}

	std::string input = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());

	parser = Parser(reg, file, input);
}

Interpreter::Interpreter(const fs::path& dir, const std::string& input, const nlohmann::json& config):
	reg(new Register()),
	parser(reg, dir, input),
	config(config)
{}

int Interpreter::run() {
	auto program = parser.parse();
	VisitorSemantic semantic(reg, config);

	env->setup(); //prepare the environment
	semantic.visit(program);

	VisitorInterpreter runner(reg, config);
	runner.visit(program);
	return 0;
}
