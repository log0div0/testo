
#include "ModeRequestLicense.hpp"
#include <iostream>
#include <license/License.hpp>
#include <nn/OnnxRuntime.hpp>

#ifdef USE_CUDA

int request_license_mode(const RequestLicenseModeArgs& args) {
	std::cout << "Checking that the system meets the requirements ..." << std::endl;
	nn::OnnxRuntime runtime;
	runtime.selftest();
	auto info = runtime.get_device_info();
	std::cout << info.name << std::endl;
	std::cout << info.uuid_str << std::endl;
	std::cout << "Everything is OK" << std::endl;
	return 0;
}

#endif
