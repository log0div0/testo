
#include "error.hpp"
#include "string.hpp"
#include <sstream>

namespace vbox {

struct ErrorInfo {
	ErrorInfo() {
		HRESULT rc = api->pfnGetException(&handle);
		if (FAILED(rc)) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
	}
	~ErrorInfo() {
		if (handle) {
			IErrorInfo_Release(handle);
		}
	}

	operator bool() const {
		return handle != nullptr;
	}

	std::string message() const {
		BSTR result = nullptr;
		HRESULT rc = IErrorInfo_get_Message(handle, &result);
		if (FAILED(rc)) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
		return StringOut(result);
	}

	IErrorInfo* handle = nullptr;
};

Error::Error(HRESULT rc) {
	try {
		ErrorInfo error_info;
		if (error_info) {
			_what = error_info.message();
		} else {
			std::stringstream ss;
			ss << "Error code: " << std::hex << rc << std::dec;
			_what = ss.str();
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Error::~Error() {}

const char* Error::what() const noexcept {
	return _what.c_str();
}

}
