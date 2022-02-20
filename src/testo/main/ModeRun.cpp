
#include "ModeRun.hpp"
#include "../IR/Program.hpp"
#include "../parser/Parser.hpp"
#include "../Utils.hpp"
#include "../Logger.hpp"

void RunModeArgs::validate() const {
	ProgramConfig::validate();
}

int run_mode(const RunModeArgs& args) {
	TRACE();

	args.validate();
	auto parser = Parser::load(args.target);
	auto ast = parser.parse();
	IR::Program program(ast, args);
	program.validate();
	program.run();

	return 0;
}
