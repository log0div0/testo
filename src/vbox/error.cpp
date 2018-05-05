
#include "error.hpp"
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

	IErrorInfo* handle = nullptr;
};

Error::Error(HRESULT rc) {
	try {
		ErrorInfo error_info;
		if (error_info) {
			_what = "TODO";
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
