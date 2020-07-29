
#pragma once

#include <vector>
#include <string>

struct RunModeArgs {
	std::string target;
	std::string prefix;
	std::string test_spec;
	std::string exclude;
	std::string invalidate;
	std::string report_folder;
	std::string license;

	std::vector<std::string> params_names;
	std::vector<std::string> params_values;

	bool stop_on_fail = false;
	bool assume_yes = false;
	bool report_logs = false;
	bool report_screenshots = false;
	bool html = false;
};

int run_mode(const RunModeArgs& args);