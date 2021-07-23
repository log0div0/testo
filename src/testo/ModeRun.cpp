
#include "ModeRun.hpp"
#include "GetDeviceInfo.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"
#include "Utils.hpp"

void RunModeArgs::validate() const {
	IR::ProgramConfig::validate();
}

int run_mode(const RunModeArgs& args) {
	args.validate();
	auto parser = Parser::load(args.target);
	auto ast = parser.parse();
	IR::Program program(ast, args);
	program.validate();
	program.run();

	return 0;
}
