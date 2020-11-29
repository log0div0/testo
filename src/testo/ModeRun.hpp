
#pragma once

#include <vector>
#include <string>
#include "IR/Program.hpp"

struct RunModeArgs: IR::ProgramConfig {
	std::string license;
	void validate() const;
};

int run_mode(const RunModeArgs& args);