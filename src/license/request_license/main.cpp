
#include <iostream>
#include <clipp.h>

#include <ghc/filesystem.hpp>

#include "../GetDeviceInfo.hpp"
#include "../License.hpp"
#include "../../nn/OnnxRuntime.hpp"

namespace fs = ghc::filesystem;

int request_license_mode(const std::string& output_path) {
	auto path_to_save = fs::absolute(output_path).generic_string();
	if (fs::exists(path_to_save)) {
		throw std::runtime_error("Error: The file \"" + path_to_save + "\" already exists");
	}
	std::cout << "Checking that the system meets the requirements ..." << std::endl;
	nn::onnx::Runtime runtime;
	runtime.selftest();
	auto info = GetDeviceInfo(0);
	nlohmann::json request = {
		{"device_name", info.name},
		{"device_uuid", info.uuid_str}
	};
	std::string container = license::pack(request, "QpCtaVw2J4EsVcBlp0KH5VA5GWXpHMdpTQqR8vCoMzIrh8MiA8wr8X8IWi4mbhvjLRLrS8QIs6Gw0ZiSmQXIBA==");
	license::write_file(path_to_save, container);
	std::cout << "Everything is OK" << std::endl;
	std::cout << "The request is saved to the file \"" << path_to_save << "\"" << std::endl;
	std::cout << "Please upload this file to the payment form on the site https://testo-lang.ru" << std::endl;
	return 0;
}

int main(int argc, char** argv) {
	try {

#ifndef USE_CUDA
		std::cout << "Request license is only applicable to the commercial GPU-version of testo. Please install appropriate distribution first\n";
		return 0;
#endif

		std::string output_path;
		auto cli = clipp::group(
			clipp::option("--out") & clipp::value("The output file for the license request", output_path)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		if (!output_path.length()) {
			output_path = "testo_license_request";
		}

		return request_license_mode(output_path);

	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}

	return 0;
}
