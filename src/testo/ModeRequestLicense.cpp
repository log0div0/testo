
#include "ModeRequestLicense.hpp"
#include <license/License.hpp>
#include <nn/OnnxRuntime.hpp>
#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#ifdef USE_CUDA

int request_license_mode(const RequestLicenseModeArgs& args) {
	std::cout << "Checking that the system meets the requirements ..." << std::endl;
	nn::OnnxRuntime runtime;
	runtime.selftest();
	auto info = nn::GetDeviceInfo();
	nlohmann::json request = {
		{"device_name", info.name},
		{"device_uuid", info.uuid_str}
	};
	std::string container = license::pack(request, "QpCtaVw2J4EsVcBlp0KH5VA5GWXpHMdpTQqR8vCoMzIrh8MiA8wr8X8IWi4mbhvjLRLrS8QIs6Gw0ZiSmQXIBA==");
	auto path_to_save = fs::absolute(args.out).generic_string();
	if (fs::exists(path_to_save)) {
		throw std::runtime_error("The path " + path_to_save + " already exists");
	}
	license::write_file(path_to_save, container);
	std::cout << "Everything is OK" << std::endl;
	std::cout << "The request is saved to the file " << path_to_save << std::endl;
	std::cout << "Please upload this file to the payment form on the page https://testo-lang.ru/sales" << std::endl;
	return 0;
}

#endif