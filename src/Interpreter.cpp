
#include <Interpreter.hpp>
#include <VisitorInterpreter.hpp>
#include <VisitorSemantic.hpp>

using namespace AST;

Interpreter::Interpreter(const fs::path& file):
	global(),
	parser(file)
{}

int Interpreter::run() {
	auto program = parser.parse();
	//VisitorSemantic semantic(global);

	//global.setup(); //prepare the environment
	//semantic.visit(program);

	//VisitorInterpreter runner(global);
	//runner.visit(program);
	return 0;
}
