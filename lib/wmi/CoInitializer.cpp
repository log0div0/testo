
#include "CoInitializer.hpp"
#include "Error.hpp"
#include <Windows.h>
#include <comdef.h>

namespace wmi {

CoInitializer::CoInitializer() {
	try {
		throw_if_failed(CoInitializeEx(0, COINIT_MULTITHREADED));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

CoInitializer::~CoInitializer() {
	CoUninitialize();
}

void CoInitializer::initalize_security() {
	try {
		throw_if_failed(CoInitializeSecurity(
			nullptr,
			-1,
			nullptr,
			nullptr,
			RPC_C_AUTHN_LEVEL_DEFAULT,
			RPC_C_IMP_LEVEL_IMPERSONATE,
			nullptr,
			EOAC_NONE,
			NULL
		));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
