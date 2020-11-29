
#include "ModeRun.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"
#include "Utils.hpp"

#ifdef USE_CUDA
#include <license/License.hpp>
#include <nn/OnnxRuntime.hpp>

void verify_license(const std::string& path_to_license) {
	if (!fs::exists(path_to_license)) {
		throw std::runtime_error("File " + path_to_license + " does not exists");
	}

	std::string container = license::read_file(path_to_license);
	nlohmann::json license = license::unpack(container, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");

	license::Date not_before(license.at("not_before").get<std::string>());
	license::Date not_after(license.at("not_after").get<std::string>());
	license::Date now(std::chrono::system_clock::now());
	license::Date release_date(TESTO_RELEASE_DATE);

	if (now < release_date) {
		throw std::runtime_error("System time is incorrect");
	}

	if (now < not_before) {
		throw std::runtime_error("The license period has not yet come");
	}

	if (now > not_after) {
		throw std::runtime_error("The license period has already ended");
	}

	auto info = nn::onnx::GetDeviceInfo();

	std::string device_uuid = license.at("device_uuid");
	if (info.uuid_str != device_uuid) {
		throw std::runtime_error("The graphics accelerator does not match the one specified in the license");
	}
}
#endif

void RunModeArgs::validate() const {
#ifdef USE_CUDA
	if (license.size()) {
		verify_license(license);
	} else {
		throw std::runtime_error("To start the program you must specify the path to the license file (--license argument)");
	}
#endif

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
