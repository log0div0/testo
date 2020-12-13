
#include "WbemLocator.hpp"
#include "Error.hpp"
#include <comutil.h>

namespace wmi {

WbemLocator::WbemLocator() {
	try {
		throw_if_failed(CoCreateInstance(
			CLSID_WbemLocator,
			0,
			CLSCTX_INPROC_SERVER,
			IID_IWbemLocator, (LPVOID *)&handle));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemServices WbemLocator::connectServer(const std::string& path) {
	try {
		IWbemServices* services = nullptr;

		throw_if_failed(handle->ConnectServer(
			bstr_t(path.c_str()),
			nullptr,
			nullptr,
			L"MS_409",
			0,
			0,
			0,
			&services
		));

		return services;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}

