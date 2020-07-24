
#pragma once

#include <string>

#ifdef USE_CUDA

struct RequestLicenseModeArgs {
	std::string out;
};

int request_license_mode(const RequestLicenseModeArgs& args);

#endif
