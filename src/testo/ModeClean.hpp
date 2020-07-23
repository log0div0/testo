
#pragma once

#include <string>

struct CleanModeArgs {
	std::string prefix;
};

int clean_mode(const CleanModeArgs& args);
