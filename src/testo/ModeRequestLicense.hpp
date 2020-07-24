
#pragma once

#include <string>

#ifdef USE_CUDA

struct RequestLicenseModeArgs {
	std::string out = "testo_license_request";
};

int request_license_mode(const RequestLicenseModeArgs& args);

#endif
