
#pragma once

#include <vector>
#include <string>
#include "Configs.hpp"

struct RunModeArgs: ProgramConfig {
	void validate() const;
};

int run_mode(const RunModeArgs& args);