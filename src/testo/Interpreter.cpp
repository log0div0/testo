
#include "Interpreter.hpp"
#include "VisitorInterpreter.hpp"
#include "VisitorSemantic.hpp"

using namespace AST;

Interpreter::Interpreter(Environment& env, const fs::path& file):
	env(env),
	parser(file)
{}

int Interpreter::run() {
	auto program = parser.parse();
	VisitorSemantic semantic(reg, env);

	env.setup(); //prepare the environment
	semantic.visit(program);

	VisitorInterpreter runner(reg);
	runner.visit(program);
	return 0;
}
