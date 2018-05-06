
#include "throw_if_failed.hpp"
#include "string.hpp"
#include "virtual_box_error_info.hpp"
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

	ErrorInfo(const ErrorInfo&) = delete;
	ErrorInfo& operator=(const ErrorInfo&) = delete;
	ErrorInfo(ErrorInfo&& other);
	ErrorInfo& operator=(ErrorInfo&& other);

	operator bool() const {
		return handle != nullptr;
	}

	template <typename IID>
	void* query_interface(IID&& iid) const {
		void* result = nullptr;
#ifdef WIN32
		HRESULT rc = IErrorInfo_QueryInterface(handle, iid, &result);
#else
		HRESULT rc = IErrorInfo_QueryInterface(handle, &iid, &result);
#endif
		if (FAILED(rc)) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
		return result;
	}

	IErrorInfo* handle = nullptr;
};

void throw_if_failed(HRESULT rc) {
	try {
		if (FAILED(rc)) {
			ErrorInfo error_info;
			if (error_info) {
				VirtualBoxErrorInfo virtual_box_error_info = (IVirtualBoxErrorInfo*)error_info.query_interface(IID_IVirtualBoxErrorInfo);
				throw std::runtime_error(virtual_box_error_info.text());
			} else {
				std::stringstream ss;
				ss << "Error code: " << std::hex << rc << std::dec;
				throw std::runtime_error(ss.str());
			}
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
