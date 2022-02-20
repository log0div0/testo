
#pragma once

#include <string>

struct CleanModeArgs {
	std::string prefix;
	bool assume_yes = false;
};

int clean_mode(const CleanModeArgs& args);
