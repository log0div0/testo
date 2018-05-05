
#include "error.hpp"
#include "string.hpp"
#include <sstream>

namespace vbox {

struct VirtualBoxErrorInfo {
	VirtualBoxErrorInfo(IVirtualBoxErrorInfo* handle): handle(handle) {
		if (!handle) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
	}

	~VirtualBoxErrorInfo() {
		if (handle) {
			IVirtualBoxErrorInfo_Release(handle);
		}
	}

	std::string text() const {
		BSTR result = nullptr;
		HRESULT rc = IVirtualBoxErrorInfo_get_Text(handle, &result);
		if (FAILED(rc)) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
		return StringOut(result);
	}

	VirtualBoxErrorInfo(const VirtualBoxErrorInfo&) = delete;
	VirtualBoxErrorInfo& operator=(const VirtualBoxErrorInfo&) = delete;
	VirtualBoxErrorInfo(VirtualBoxErrorInfo&& other);
	VirtualBoxErrorInfo& operator=(VirtualBoxErrorInfo&& other);

	IVirtualBoxErrorInfo* handle;
};

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

	ErrorInfo(const ErrorInfo&) = delete;
	ErrorInfo& operator=(const ErrorInfo&) = delete;
	ErrorInfo(ErrorInfo&& other);
	ErrorInfo& operator=(ErrorInfo&& other);

	operator bool() const {
		return handle != nullptr;
	}

	void* query_interface(const IID& iid) const {
		void* result = nullptr;
		HRESULT rc = IErrorInfo_QueryInterface(handle, iid, &result);
		if (FAILED(rc)) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
		return result;
	}

	IErrorInfo* handle = nullptr;
};

Error::Error(HRESULT rc) {
	try {
		ErrorInfo error_info;
		if (error_info) {
			VirtualBoxErrorInfo virtual_box_error_info = (IVirtualBoxErrorInfo*)error_info.query_interface(IID_IVirtualBoxErrorInfo);
			_what = virtual_box_error_info.text();
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
