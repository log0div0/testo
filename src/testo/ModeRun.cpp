
#include "ModeRun.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"
#include "Utils.hpp"

#ifdef USE_CUDA
#include <license/License.hpp>
#endif


int run_mode(const RunModeArgs& args) {

#ifdef USE_CUDA
	if (args.license.size()) {
		verify_license(args.license, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");
	} else {
		throw std::runtime_error("To start the program you must specify the path to the license file (--license argument)");
	}
#endif

	auto params = nlohmann::json::array();

	std::set<std::string> unique_param_names;

	for (size_t i = 0; i < args.params_names.size(); ++i) {
		auto result = unique_param_names.insert(args.params_names[i]);
		if (!result.second) {
			throw std::runtime_error("Error: param \"" + args.params_names[i] + "\" is defined multiple times as a command line argument");
		}
		nlohmann::json json_param = {
			{ "name", args.params_names[i]},
			{ "value", args.params_values[i]}
		};
		params.push_back(json_param);
	}

	nlohmann::json config = {
		{"stop_on_fail", args.stop_on_fail},
		{"assume_yes", args.assume_yes},
		{"test_spec", args.test_spec},
		{"exclude", args.exclude},
		{"invalidate", args.invalidate},
		{"report_folder", args.report_folder},
		{"report_logs", args.report_logs},
		{"report_screenshots", args.report_screenshots},
		{"html", args.html},
		{"prefix", args.prefix},
		{"params", params}
	};

	if (!fs::exists(args.target)) {
		throw std::runtime_error(std::string("Fatal error: target doesn't exist: ") + args.target);
	}

	auto parser = Parser::load(args.target);
	auto ast = parser.parse();
	IR::Program program(ast, config);
	program.validate();
	program.run();

	return 0;
}
